"""Regresión logística WoE 5.0. Selección exclusivamente TRAIN, con trazabilidad."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning
import shap
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

SEGMENTS=('TRAIN','TEST','OOT')
ROOT=Path('5.MODELO_LOGISTICO')
MULT_ROOT=Path('4.ANALIS_UV_BV_MV/MULTIVARIADO')
BIV_ROOT=Path('4.ANALIS_UV_BV_MV/BIVARIADO')
DEFAULTS=dict(P_VALUE_THRESHOLD=.05,SHAP_MIN_PCT=2.0,MAX_IV_MODEL=.50,
    MAX_ITERACIONES=100,CLASSIFICATION_THRESHOLD=.50,RANDOM_STATE=42,
    SHAP_SAMPLE_N=2000,SHAP_BACKGROUND_N=500,FIT_MAXITER=200,
    MAX_CONDITION_NUMBER=1e12,GINI_ALERT_TOLERANCE=.05,KS_ALERT_TOLERANCE=.05)


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration(overrides=None):
    config={**DEFAULTS,**(overrides or {})}
    if set(config)!=set(DEFAULTS):raise ValueError('Parámetro desconocido')
    for key in ('P_VALUE_THRESHOLD','MAX_IV_MODEL','CLASSIFICATION_THRESHOLD','GINI_ALERT_TOLERANCE','KS_ALERT_TOLERANCE'):
        if not np.isfinite(config[key]) or not 0<=config[key]<=1:raise ValueError(f'Parámetro inválido: {key}')
    if not 0<=config['SHAP_MIN_PCT']<=100:raise ValueError('SHAP_MIN_PCT inválido')
    for key in ('MAX_ITERACIONES','SHAP_SAMPLE_N','SHAP_BACKGROUND_N','FIT_MAXITER'):
        if not isinstance(config[key],int) or config[key]<1:raise ValueError(f'Parámetro entero requerido: {key}')
    return config


def import_bivariado():
    spec=importlib.util.spec_from_file_location('bivariado_modelo','3.0.0.bivariado.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def load_inputs(mult_root=MULT_ROOT,biv_root=BIV_ROOT):
    """Carga CSV finales existentes; no transforma ni recalcula cortes/WoE/correlación."""
    mult_root,biv_root=Path(mult_root),Path(biv_root)
    biv=import_bivariado();bundle=biv.load_v4(root=biv_root)
    artifact_path=biv_root/'binning_woe_train.json'
    artifact=json.loads(artifact_path.read_text(encoding='utf-8'))
    selection=json.loads((mult_root/'seleccion_correlacion.json').read_text(encoding='utf-8'))
    manifest=json.loads((mult_root/'matrices/manifest_indices.json').read_text(encoding='utf-8'))
    if selection['woe_artifact_sha256']!=sha256(artifact_path) or selection['learned_on']!='TRAIN':
        raise ValueError('Selección 4.0 y hash WoE incompatibles')
    matrices={};rows=[];features=selection['retained_variables']
    if not features or len(features)!=len(set(features)):raise ValueError('Selección duplicada o vacía')
    for segment in SEGMENTS:
        file=mult_root/f'matrices/X_{segment.lower()}_uncorr.csv';entry=manifest[segment]
        if sha256(file)!=entry['sha256']:raise ValueError(f'{segment}: hash CSV multivariado alterado')
        X=pd.read_csv(file,index_col='_row_id',float_precision='round_trip')
        X.index.name=bundle['X'][segment].index.name
        y=bundle['y'][segment]
        if (X.columns.has_duplicates or X.index.has_duplicates or X.columns.tolist()!=features
                or not X.index.equals(y.index) or not X.index.equals(bundle['X'][segment].index)
                or X.index.tolist()!=entry['row_ids'] or len(X)!=entry['n_rows']
                or str(X.index.dtype)!=entry['index_dtype'] or y.tolist()!=entry['target']
                or {c:str(d) for c,d in X.dtypes.items()}!=entry['dtypes']):
            raise ValueError(f'{segment}: identidad, orden, variables o tipos incompatibles')
        if y.name!=artifact['target'] or not y.isin([0,1]).all() or y.isna().any():
            raise ValueError(f'{segment}: target no corresponde a la definición B.3')
        pd.testing.assert_series_equal(y,bundle['metadata'][segment].indicador_default,check_exact=True)
        if not bundle['metadata'][segment].SEGMENTO.eq(segment).all():raise ValueError('Segmento alterado')
        if not all(pd.api.types.is_numeric_dtype(d) for d in X.dtypes) or not np.isfinite(X.to_numpy(float)).all():
            raise ValueError(f'{segment}: WoE no numérico/finito; detener sin imputación')
        # Comparar con los WoE persistidos B.3; no invocar transform_woe.
        old=pd.read_csv(biv_root/f'matrices/X_{segment.lower()}_woe.csv',index_col=0,float_precision='round_trip')
        old.index.name=X.index.name
        pd.testing.assert_frame_equal(X,old.loc[:,features],check_exact=True)
        matrices[segment]=X
        rows.append(dict(Segmento=segment,Registros=len(X),Variables=len(features),Target=y.name,
            Default_1=int(y.sum()),Estado='VALIDADO'))
    iv=pd.read_csv(mult_root/'reportes/df_iv_final.csv',float_precision='round_trip')
    if iv.Variable.duplicated().any() or not set(features).issubset(iv.Variable):raise ValueError('IV multivariado incompatible')
    for row in iv.itertuples():
        cfg=artifact['variables'][row.Variable]
        if cfg['state']!='APROBADA' or not np.isclose(row.IV_final,cfg['iv_total'],rtol=1e-12,atol=1e-14):
            raise ValueError(f'IV final no corresponde a cortes congelados: {row.Variable}')
    expected=iv.loc[iv.Variable.isin(features)].sort_values(['IV_final','Variable'],ascending=[False,True]).Variable.tolist()
    if features!=expected:raise ValueError('Orden final 4.0 no corresponde al ranking IV')
    return dict(X=matrices,y=bundle['y'],bundle=bundle,artifact=artifact,selection=selection,
        iv=iv,validation=pd.DataFrame(rows),binning_path=artifact_path)


def validate_woe_convention(artifact,source):
    """Valida metadatos y expresión AST original, sin recalcular ningún WoE."""
    if artifact.get('target')!='indicador_default' or artifact.get('target_definition')!={'0':'bueno','1':'malo'}:
        raise ValueError('Target 1=default no comprobado en artefacto actual')
    expressions=[]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if (isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name)
                        and target.value.id=='stats' and isinstance(target.slice,ast.Constant) and target.slice.value=='WoE'):
                    expressions.append(ast.unparse(node.value).replace(' ',''))
    definition=artifact.get('woe_definition','')
    good='np.log(stats.Dist_buenos_suavizada/stats.Dist_malos_suavizada)'
    bad='np.log(stats.Dist_malos_suavizada/stats.Dist_buenos_suavizada)'
    if expressions==[good] and definition.startswith('ln(P(bin|good)/P(bin|bad))'):
        sign=-1;formula='ln(%Buenos suavizados / %Malos suavizados)'
    elif expressions==[bad] and definition.startswith('ln(P(bin|bad)/P(bin|good))'):
        sign=1;formula='ln(%Malos suavizados / %Buenos suavizados)'
    else:raise ValueError('Convención WoE desconocida o inconsistente entre código y JSON')
    return dict(target='indicador_default',default_value=1,formula=formula,expected_sign=sign,
        source_expression=expressions[0],definition=definition,intercept_exempt=True)


def filter_iv(iv,features,max_iv=.50):
    if len(features)!=len(set(features)) or iv.Variable.duplicated().any() or not set(features).issubset(iv.Variable):
        raise ValueError('IV o variables incompatibles')
    table=iv.set_index('Variable').loc[features].reset_index()
    if not np.isfinite(table.IV_final).all() or table.IV_final.lt(0).any():raise ValueError('IV inválido')
    table['Estado_IV']=np.where(table.IV_final.gt(max_iv),'EXCLUIDA_IV','INCLUIDA')
    table['Motivo']=np.where(table.IV_final.gt(max_iv),f'IV > {max_iv}: alerta de revisión por política; no demuestra fuga','IV dentro de política')
    kept=table.loc[table.Estado_IV.eq('INCLUIDA'),'Variable'].tolist()
    if not kept:raise ValueError('Sin predictores tras filtro IV')
    return kept,table


def design_matrix(X):
    if 'const' in X.columns or X.columns.has_duplicates:raise ValueError('Predictor const reservado/duplicado')
    return sm.add_constant(X.astype(float),has_constant='add')


class ModelValidationError(RuntimeError):
    """Error explícito de ajuste/política; contiene estado para recuperar el historial."""
    def __init__(self,message,state=None):super().__init__(message);self.state=state


def fit_logit(X,y,config):
    if not X.index.equals(y.index) or X.index.has_duplicates:raise ModelValidationError('Índices TRAIN desalineados')
    if X.empty or y.nunique()!=2 or not y.isin([0,1]).all():raise ModelValidationError('TRAIN sin predictores o sin ambas clases')
    D=design_matrix(X)
    if not np.isfinite(D.to_numpy()).all():raise ModelValidationError('WoE TRAIN no finito')
    if np.linalg.matrix_rank(D.to_numpy())!=D.shape[1]:raise ModelValidationError('Matriz singular: rango incompleto')
    condition=float(np.linalg.cond(D.to_numpy()))
    if condition>config['MAX_CONDITION_NUMBER']:raise ModelValidationError(f'Diseño numéricamente inestable: {condition}')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:result=sm.Logit(y,D).fit(disp=False,maxiter=config['FIT_MAXITER'])
        except (PerfectSeparationError,np.linalg.LinAlgError,ValueError) as exc:
            raise ModelValidationError(f'Fallo Logit: {type(exc).__name__}: {exc}') from exc
    warning_text=[f'{w.category.__name__}: {w.message}' for w in caught]
    if any(issubclass(w.category,PerfectSeparationWarning) for w in caught):
        raise ModelValidationError('Separación perfecta detectada: '+'; '.join(warning_text))
    if not result.mle_retvals.get('converged',False):raise ModelValidationError('Logit sin convergencia: '+'; '.join(warning_text))
    if not np.isfinite(np.r_[result.params,result.bse,result.pvalues,result.conf_int().to_numpy().ravel()]).all():
        raise ModelValidationError('Inferencia no finita: separación/singularidad posible')
    hessian=-result.model.hessian(result.params)
    hcondition=float(np.linalg.cond(hessian))
    if hcondition>config['MAX_CONDITION_NUMBER'] or np.linalg.eigvalsh(hessian).min()<=0:
        raise ModelValidationError(f'Hessiano singular/inestable: {hcondition}')
    return result,dict(Condition_design=condition,Condition_hessian=hcondition,Advertencias='; '.join(warning_text))


def coefficient_table(model):
    ci=model.conf_int()
    return pd.DataFrame(dict(Variable=model.params.index,Coeficiente=model.params.to_numpy(),
        Error_estandar=model.bse.to_numpy(),z=model.tvalues.to_numpy(),P_value=model.pvalues.to_numpy(),
        IC_025=ci.iloc[:,0].to_numpy(),IC_975=ci.iloc[:,1].to_numpy()))


def linear_shap(model,X_train,config):
    """SHAP interventional lineal, en log-odds, con coeficientes del MISMO Logit."""
    features=model.params.index.drop('const').tolist()
    if X_train.columns.tolist()!=features:raise ValueError('SHAP no corresponde al orden del Logit')
    sample=X_train.sample(n=min(len(X_train),config['SHAP_SAMPLE_N']),random_state=config['RANDOM_STATE'])
    background=X_train.sample(n=min(len(X_train),config['SHAP_BACKGROUND_N']),random_state=config['RANDOM_STATE'])
    coef=model.params.loc[features].to_numpy();intercept=float(model.params['const'])
    masker=shap.maskers.Independent(background,max_samples=len(background))
    explainer=shap.LinearExplainer((coef,intercept),masker)
    explanation=explainer(sample)
    values=np.asarray(explanation.values,dtype=float)
    expected=intercept+float(background.mean().to_numpy()@coef)
    # Verificar identidad analítica y aditividad en la escala logit, sin segundo ajuste.
    analytical=(sample.to_numpy()-background.mean().to_numpy())*coef
    if not np.allclose(values,analytical,rtol=1e-10,atol=1e-12):raise ModelValidationError('SHAP lineal inconsistente con coeficientes')
    if not np.allclose(expected+values.sum(axis=1),design_matrix(sample).to_numpy()@model.params.to_numpy(),rtol=1e-10,atol=1e-12):
        raise ModelValidationError('SHAP no reconstruye el logit del modelo vigente')
    mean_abs=np.abs(values).mean(axis=0);total=float(mean_abs.sum())
    if not np.isfinite(total) or total<=0:raise ModelValidationError('Importancia SHAP total nula/indefinida')
    table=pd.DataFrame(dict(Variable=features,Mean_abs_SHAP=mean_abs,SHAP_pct=100*mean_abs/total))
    table=table.sort_values(['SHAP_pct','Variable'],ascending=[False,True]).reset_index(drop=True)
    return dict(table=table,values=pd.DataFrame(values,index=sample.index,columns=features),
        sample=sample,background_indices=background.index.tolist(),expected_value=expected,
        scale='log-odds',method='shap.LinearExplainer; interventional independent TRAIN',
        coefficients=model.params.to_dict())


class SeleccionIterativa:
    def __init__(self,X_train,y_train,expected_sign,config=None):
        self.config=configuration(config);self.X=X_train.copy(deep=True);self.y=y_train.copy(deep=True)
        if expected_sign not in (-1,1):raise ValueError('Signo esperado debe ser -1 o +1')
        self.expected_sign=expected_sign;self.variables=X_train.columns.tolist()
        self.history=[];self.coefficients=[];self.shap_history=[];self.model=None;self.shap=None
        self.status='INICIADO';self._fit()

    def _fail(self,message):
        self.status='DETENIDO_INVALIDO'
        if self.history:self.history[-1]['Condicion_parada']=message
        raise ModelValidationError(message,self)

    def _fit(self):
        iteration=len(self.history)+1
        try:self.model,diagnostics=fit_logit(self.X[self.variables],self.y,self.config)
        except ModelValidationError as exc:
            self.history.append(dict(Iteracion=iteration,N_variables=len(self.variables),
                Estado_convergencia=False,Criterio_eliminacion='ERROR_AJUSTE',Variable_eliminada=None,
                Condicion_parada=str(exc)))
            self._fail(str(exc))
        self.shap=None
        model=self.model
        self.history.append(dict(Iteracion=iteration,N_variables=len(self.variables),Variable_eliminada=None,
            Criterio_eliminacion='PENDIENTE',Coeficiente=np.nan,P_value=np.nan,SHAP_pct=np.nan,
            AIC=float(model.aic),BIC=float(model.bic),Log_Likelihood=float(model.llf),
            Pseudo_R2=float(model.prsquared),Estado_convergencia=True,Condicion_parada='',**diagnostics))
        self.coefficients.append(coefficient_table(model).assign(Iteracion=iteration))

    def _candidate(self,stage):
        coef=self.model.params.drop('const');pvalues=self.model.pvalues.drop('const')
        wrong=coef[coef*self.expected_sign<0]
        if len(wrong):
            # Mayor magnitud de signo contrario, luego nombre alfabético.
            var=sorted(wrong.index,key=lambda v:(-abs(wrong[v]),v))[0]
            return var,'SIGNO'
        if stage=='signo':return None,'SIGNOS_VALIDOS_PENDIENTE_PVALUE'
        bad=pvalues[pvalues>self.config['P_VALUE_THRESHOLD']]
        if len(bad):return sorted(bad.index,key=lambda v:(-bad[v],v))[0],'P_VALUE'
        if stage=='pvalue':return None,'SIGNOS_PVALUES_VALIDOS_PENDIENTE_SHAP'
        self.shap=linear_shap(self.model,self.X[self.variables],self.config)
        self.shap_history.append(self.shap['table'].assign(Iteracion=len(self.history)))
        bad=self.shap['table'].loc[self.shap['table'].SHAP_pct.lt(self.config['SHAP_MIN_PCT'])]
        if len(bad):return bad.sort_values(['SHAP_pct','Variable']).iloc[0].Variable,'SHAP'
        return None,'POLITICA_CUMPLIDA'

    def run(self,stage='completo'):
        if stage not in ('signo','pvalue','shap','completo'):raise ValueError('Etapa desconocida')
        while True:
            try:var,criterion=self._candidate(stage)
            except ModelValidationError as exc:self._fail(str(exc))
            row=self.history[-1]
            if var is None:
                row['Criterio_eliminacion']='CONSERVAR';row['Condicion_parada']=criterion
                self.status='VALIDADO' if criterion=='POLITICA_CUMPLIDA' else criterion
                break
            if len(self.variables)==1:self._fail(f'Último predictor incumple {criterion}; no se permite modelo vacío')
            if len(self.history)>=self.config['MAX_ITERACIONES']:self._fail('MAX_ITERACIONES alcanzado; política no certificada')
            row.update(Variable_eliminada=var,Criterio_eliminacion=criterion,
                Coeficiente=float(self.model.params[var]),P_value=float(self.model.pvalues[var]),
                SHAP_pct=float(self.shap['table'].set_index('Variable').loc[var,'SHAP_pct']) if criterion=='SHAP' else np.nan,
                Condicion_parada='ELIMINAR_UNA_Y_REAJUSTAR')
            self.variables.remove(var);self._fit()
        return self

    @property
    def df_iteraciones_modelo(self):return pd.DataFrame(self.history)

    @property
    def df_coeficientes_iteraciones(self):return pd.concat(self.coefficients,ignore_index=True) if self.coefficients else pd.DataFrame()


def eliminar_variables_completo(X_train,y_train,expected_sign,config=None,estado=None,etapa='completo'):
    state=estado or SeleccionIterativa(X_train,y_train,expected_sign,config)
    pd.testing.assert_frame_equal(state.X,X_train,check_exact=True)
    pd.testing.assert_series_equal(state.y,y_train,check_exact=True)
    if state.expected_sign!=expected_sign or state.config!=configuration(config):raise ValueError('Estado y configuración incompatibles')
    return state.run(etapa)


def predict_segments(model,matrices,features):
    if model.params.index.tolist()!=['const']+list(features):raise ValueError('Modelo y orden de predictores incompatibles')
    predictions={}
    for s in SEGMENTS:
        X=matrices[s]
        if X.columns.tolist()!=list(features) or not np.isfinite(X.to_numpy(float)).all():
            raise ValueError(f'{s}: columnas o valores incompatibles')
        proba=pd.Series(model.predict(design_matrix(X)),index=X.index,name='Probabilidad_default')
        if not np.isfinite(proba).all() or not proba.between(0,1).all():raise ValueError('Probabilidades inválidas')
        predictions[s]=proba
    return predictions


def metric_arrays(y,proba):
    if isinstance(y,pd.Series) and isinstance(proba,pd.Series) and not y.index.equals(proba.index):
        raise ValueError('Métricas: índices desalineados')
    y,p=np.asarray(y),np.asarray(proba,dtype=float)
    if len(y)!=len(p) or not len(y) or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Target/probabilidades inválidos')
    return y,p


def gini_auc(y,proba):
    y,p=metric_arrays(y,proba)
    if np.unique(y).size<2:return np.nan,np.nan
    auc=float(roc_auc_score(y,p));return 2*auc-1,auc


def ks_stat(y,proba):
    y,p=metric_arrays(y,proba)
    if np.unique(y).size<2:return np.nan
    fpr,tpr,_=roc_curve(y,p);return float(np.max(tpr-fpr))


def calcular_metricas(y,proba,threshold=.50):
    y,p=metric_arrays(y,proba)
    if not 0<=threshold<=1:raise ValueError('Umbral inválido')
    predicted=(p>=threshold).astype(int);tn,fp,fn,tp=map(int,confusion_matrix(y,predicted,labels=[0,1]).ravel())
    gini,auc=gini_auc(y,p);ks=ks_stat(y,p);notes=[]
    recall=tp/(tp+fn) if tp+fn else np.nan
    precision=tp/(tp+fp) if tp+fp else np.nan
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan
    if np.unique(y).size<2:notes.append('AUC/Gini/KS no definidos: target sin ambas clases')
    if not tp+fn:notes.append('Recall no definido: sin defaults observados')
    if not tp+fp:notes.append('Precision no definida: sin positivos predichos')
    if not 2*tp+fp+fn:notes.append('F1 no definido: sin positivos observados ni predichos')
    return dict(N=len(y),AUC=auc,Gini=gini,KS=ks,Recall=recall,Precision=precision,F1_score=f1,
        Brier=float(brier_score_loss(y,p)),Accuracy=float((predicted==y).mean()),
        Tasa_default_observada=float(y.mean()),Probabilidad_promedio=float(p.mean()),
        TN=tn,FP=fp,FN=fn,TP=tp,Umbral=threshold,Advertencias='; '.join(notes))


def print_metricas(metrics):
    data=metrics.set_index('Segmento')
    print('📊 Métricas del modelo final (RL + WoE):\n'+'-'*50)
    for s in SEGMENTS:
        row=data.loc[s];print(f'  {s:<8} → AUC: {row.AUC:.4f} | Gini: {row.Gini:.4f} | KS: {row.KS:.4f}')
    for s in ('TEST','OOT'):
        delta=data.loc['TRAIN','Gini']-data.loc[s,'Gini']
        print(f'  Δ Gini (TRAIN - {s}): {delta:.4f} | absoluto: {abs(delta):.4f}')
    print('\n📊 Métricas de clasificación y calibración:\n'+'-'*65)
    print(metrics[['Segmento','Recall','Precision','F1_score','Brier','Accuracy']].to_string(index=False,float_format=lambda v:f'{v:.4f}'))


def segment_deltas(metrics,config):
    data=metrics.set_index('Segmento');rows=[]
    for s in ('TEST','OOT'):
        for name in ('Gini','KS'):
            delta=float(data.loc['TRAIN',name]-data.loc[s,name]);tol=config[f'{name.upper()}_ALERT_TOLERANCE']
            rows.append(dict(Comparacion=f'TRAIN - {s}',Metrica=name,Diferencia_firmada=delta,
                Diferencia_absoluta=abs(delta),Tolerancia=tol,
                Alerta='NO_EVALUABLE' if not np.isfinite(delta) else ('REVISAR' if abs(delta)>tol else 'DENTRO_TOLERANCIA'),
                Criterio='Regla descriptiva de seguimiento; no es prueba estadística'))
    return pd.DataFrame(rows)


def plot_roc(targets,predictions):
    fig,ax=plt.subplots(figsize=(7,6))
    for s in SEGMENTS:
        y,p=metric_arrays(targets[s],predictions[s])
        if np.unique(y).size<2:continue
        fpr,tpr,_=roc_curve(y,p);ax.plot(fpr,tpr,label=f'{s}: AUC={roc_auc_score(y,p):.4f}')
    ax.plot([0,1],[0,1],'k--',alpha=.5);ax.set(xlabel='Falsos positivos',ylabel='Recall',title='ROC — modelo TRAIN congelado');ax.legend();fig.tight_layout();return fig


def plot_ks(targets,predictions):
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for s,ax in zip(SEGMENTS,axes):
        y,p=metric_arrays(targets[s],predictions[s])
        if np.unique(y).size<2:ax.text(.1,.5,'KS indefinido: una clase');continue
        fpr,tpr,thresholds=roc_curve(y,p);valid=np.isfinite(thresholds)
        ax.plot(thresholds[valid],tpr[valid],label='TPR');ax.plot(thresholds[valid],fpr[valid],label='FPR')
        j=np.argmax(tpr-fpr)
        if np.isfinite(thresholds[j]):ax.vlines(thresholds[j],fpr[j],tpr[j],color='black',ls='--')
        ax.set(title=f'{s}: KS={ks_stat(y,p):.4f}',xlabel='Umbral probabilidad',ylabel='Proporción');ax.legend()
    fig.tight_layout();return fig


def plot_confusion(metrics):
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for row,ax in zip(metrics.itertuples(),axes):
        sns.heatmap([[row.TN,row.FP],[row.FN,row.TP]],annot=True,fmt='d',cmap='Blues',cbar=False,ax=ax)
        ax.set(title=f'{row.Segmento} — umbral {row.Umbral}',xlabel='Predicho (0/1)',ylabel='Observado (0/1)')
    fig.tight_layout();return fig


def plot_calibration(targets,predictions):
    fig,ax=plt.subplots(figsize=(7,6))
    for s in SEGMENTS:
        observed,predicted=calibration_curve(targets[s],predictions[s],n_bins=10,strategy='uniform')
        ax.plot(predicted,observed,'o-',label=s)
    ax.plot([0,1],[0,1],'k--');ax.set(xlabel='Probabilidad promedio del bin',ylabel='Default observado del bin',title='Calibración — 10 bins uniformes (vacíos omitidos)');ax.legend();fig.tight_layout();return fig


def plot_shap(shap_result):
    table=shap_result['table'].sort_values('SHAP_pct')
    fig,ax=plt.subplots(figsize=(12,max(5,len(table)*.4)))
    ax.barh(table.Variable,table.SHAP_pct,color='#287d8e');ax.set(xlabel='mean(|SHAP|) / total (%)',title='SHAP lineal del Logit final — TRAIN, escala log-odds')
    for i,v in enumerate(table.SHAP_pct):ax.text(v,i,f' {v:.2f}%',va='center')
    ax.margins(x=.15);fig.tight_layout();return fig


def plot_coefficients(model):
    table=coefficient_table(model);table=table.loc[table.Variable.ne('const')]
    fig,ax=plt.subplots(figsize=(12,max(5,len(table)*.4)))
    ax.errorbar(table.Coeficiente,table.Variable,xerr=np.array([table.Coeficiente-table.IC_025,table.IC_975-table.Coeficiente]),fmt='o',capsize=4)
    ax.axvline(0,color='gray',ls='--');ax.set(xlabel='Coeficiente e IC 95%',title='Logit final — coeficientes WoE');fig.tight_layout();return fig


def plot_iterations(history):
    fig,ax=plt.subplots(figsize=(9,4));ax.plot(history.Iteracion,history.N_variables,'o-')
    ax.set(xlabel='Modelo ajustado / iteración',ylabel='Número de predictores',title='Evolución de selección TRAIN')
    ax.set_xticks(history.Iteracion);fig.tight_layout();return fig


def export_results(root,state,inputs,final_matrices,predictions,metrics,deltas,iv_report,convention,figures):
    root=Path(root)
    if state.status!='VALIDADO':raise ModelValidationError('No exportar como final un modelo sin validar',state)
    for name in ('modelos','matrices','predicciones','reportes','graficos','evidencias'):(root/name).mkdir(parents=True,exist_ok=True)
    state.model.save(root/'modelos/modelo_logit.pkl')
    frames=dict(df_iteraciones_modelo=state.df_iteraciones_modelo,df_coeficientes_iteraciones=state.df_coeficientes_iteraciones,
        df_coeficientes_final=coefficient_table(state.model),df_metricas_modelo=metrics,df_deltas_modelo=deltas,
        df_filtro_iv=iv_report,variables_excluidas_iv=iv_report.loc[iv_report.Estado_IV.eq('EXCLUIDA_IV')],
        variables_finales=pd.DataFrame({'Variable':state.variables,'Orden':range(1,len(state.variables)+1)}),
        df_shap_final=state.shap['table'])
    for criterion in ('SIGNO','P_VALUE','SHAP'):
        frames[f'variables_eliminadas_{criterion.lower()}']=state.df_iteraciones_modelo.loc[state.df_iteraciones_modelo.Criterio_eliminacion.eq(criterion)]
    if state.shap_history:frames['df_shap_iteraciones']=pd.concat(state.shap_history,ignore_index=True)
    for name,frame in frames.items():frame.to_csv(root/f'reportes/{name}.csv',index=False)
    (root/'reportes/summary_statsmodels.txt').write_text(state.model.summary().as_text(),encoding='utf-8')
    coefficients=dict(intercept=float(state.model.params['const']),coefficients={v:float(state.model.params[v]) for v in state.variables},variables=state.variables)
    (root/'modelos/coeficientes.json').write_text(json.dumps(coefficients,indent=2,allow_nan=False),encoding='utf-8')
    state.shap['values'].to_csv(root/'reportes/shap_valores_train.csv',index_label='_row_id')
    (root/'evidencias/shap_muestreo.json').write_text(json.dumps(dict(
        sample_indices=state.shap['values'].index.tolist(),background_indices=state.shap['background_indices'],
        expected_value=state.shap['expected_value'],scale=state.shap['scale'],method=state.shap['method'],
        random_state=state.config['RANDOM_STATE']),indent=2),encoding='utf-8')
    for s in SEGMENTS:
        final_matrices[s].to_csv(root/f'matrices/X_{s.lower()}_modelo.csv',index_label='_row_id')
        inputs['y'][s].to_csv(root/f'matrices/y_{s.lower()}.csv',index_label='_row_id')
        pred=pd.DataFrame(dict(Target=inputs['y'][s],Probabilidad_default=predictions[s],
            Prediccion=(predictions[s]>=state.config['CLASSIFICATION_THRESHOLD']).astype(int)))
        pred.to_csv(root/f'predicciones/predicciones_{s.lower()}.csv',index_label='_row_id')
    for name,fig in figures.items():fig.savefig(root/f'graficos/{name}.png',dpi=140,bbox_inches='tight')
    references=[inputs['binning_path'],MULT_ROOT/'seleccion_correlacion.json',MULT_ROOT/'matrices/manifest_indices.json',MULT_ROOT/'reportes/df_iv_final.csv']
    config=dict(methodology_version='5.0',parameters=state.config,convention=convention,
        variables_received=inputs['selection']['retained_variables'],variables_final=state.variables,
        references={str(p):sha256(p) for p in references},learned_on='TRAIN',
        sign_rule='Mayor magnitud contraria al signo esperado; empate alfabético',
        pvalue_rule='Mayor p-value > umbral; empate alfabético',shap_rule='Menor porcentaje < umbral; empate alfabético',
        shap_scale='log-odds; LinearExplainer con coeficientes del Logit vigente',
        inference_caveat='P-values condicionados al modelo; selección iterativa puede volverlos optimistas',
        classification_caveat='0.50 es referencia inicial, no umbral operativo optimizado')
    (root/'configuracion_modelo.json').write_text(json.dumps(config,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    restored=sm.load(root/'modelos/modelo_logit.pkl');replayed=predict_segments(restored,final_matrices,state.variables)
    for s in SEGMENTS:np.testing.assert_allclose(predictions[s],replayed[s],rtol=1e-12,atol=1e-12)
    return config
