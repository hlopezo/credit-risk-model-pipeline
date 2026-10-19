"""Diagnóstico de probabilidades congeladas. No ajusta modelos ni transforma WoE."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

SEGMENTS=('TRAIN','TEST','OOT')
SOURCE=Path('5.MODELO_LOGISTICO')
ROOT=Path('5.1.EVALUACION_UMBRALES')
LIMITATION=('La validación interna es una submuestra reproducible de TRAIN, pero el modelo congelado '
    'ya se ajustó con todo TRAIN. No es validación fuera de muestra ni proporciona estimaciones '
    'independientes del ajuste. Sin reentrenar no se puede corregir esta limitación. '
    'TEST y OOT no intervienen en la propuesta de umbrales.')
DEFAULTS=dict(RANDOM_STATE=42,INTERNAL_VALIDATION_FRACTION=.30,RECALL_SCENARIO=.50,
    MAX_HIGH_RISK_SCENARIO=.10,REFERENCE_THRESHOLD=.50)


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config(overrides=None):
    cfg={**DEFAULTS,**(overrides or {})}
    if set(cfg)!=set(DEFAULTS):raise ValueError('Parámetro desconocido')
    for key in ('INTERNAL_VALIDATION_FRACTION','RECALL_SCENARIO','MAX_HIGH_RISK_SCENARIO','REFERENCE_THRESHOLD'):
        if not np.isfinite(cfg[key]) or not 0<cfg[key]<1:raise ValueError(f'Parámetro inválido: {key}')
    return cfg


def threshold_grid():return np.arange(1,51,dtype=float)/100


def validate_predictions(frame):
    if (frame.empty or frame.index.has_duplicates or not {'Target','Probabilidad_default'}.issubset(frame)
            or frame.Target.isna().any() or not frame.Target.isin([0,1]).all()
            or not np.isfinite(frame.Probabilidad_default.to_numpy(float)).all()
            or not frame.Probabilidad_default.between(0,1).all()):
        raise ValueError('Predicciones: identidad, target o probabilidades inválidos')


def load_frozen(source=SOURCE):
    source=Path(source);cfg=json.loads((source/'configuracion_modelo.json').read_text(encoding='utf-8'))
    coef=json.loads((source/'modelos/coeficientes.json').read_text(encoding='utf-8'))
    for file,digest in cfg['references'].items():
        if sha256(file)!=digest:raise ValueError(f'Hash de referencia alterado: {file}')
    model=sm.load(source/'modelos/modelo_logit.pkl')
    features=cfg['variables_final']
    if (cfg['convention']['target']!='indicador_default' or cfg['convention']['default_value']!=1
            or features!=coef['variables'] or len(features)!=9
            or model.params.index.tolist()!=['const']+features):raise ValueError('Modelo final 5.0 incompatible')
    np.testing.assert_array_equal(model.params.to_numpy(),[coef['intercept']]+[coef['coefficients'][v] for v in features])
    data={};rows=[];references=[source/'configuracion_modelo.json',source/'modelos/coeficientes.json',source/'modelos/modelo_logit.pkl']
    for segment in SEGMENTS:
        pred_path=source/f'predicciones/predicciones_{segment.lower()}.csv'
        X_path=source/f'matrices/X_{segment.lower()}_modelo.csv';y_path=source/f'matrices/y_{segment.lower()}.csv'
        pred=pd.read_csv(pred_path,index_col='_row_id',float_precision='round_trip')
        X=pd.read_csv(X_path,index_col='_row_id',float_precision='round_trip')
        y=pd.read_csv(y_path,index_col='_row_id')['indicador_default']
        validate_predictions(pred)
        if not pred.index.equals(X.index) or not pred.index.equals(y.index) or X.columns.tolist()!=features:
            raise ValueError(f'{segment}: índices/orden incompatibles')
        np.testing.assert_array_equal(pred.Target,y)
        computed=model.predict(sm.add_constant(X,has_constant='add'))
        np.testing.assert_allclose(pred.Probabilidad_default,computed,rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(pred.Prediccion,(pred.Probabilidad_default>=cfg['parameters']['CLASSIFICATION_THRESHOLD']).astype(int))
        data[segment]=pred.copy(deep=True)
        rows.append(dict(Segmento=segment,N=len(pred),Predictores=len(features),Default_1=int(y.sum()),Estado='VALIDADO',
            Probabilidades_originales_reproducidas=True))
        references.extend([pred_path,X_path,y_path])
    return dict(data=data,model=model,features=features,validation=pd.DataFrame(rows),
        references={str(p):sha256(p) for p in references},source_config=cfg)


def probability_summary(data):
    rows=[];quantiles=[.01,.05,.10,.25,.50,.75,.90,.95,.99,.995]
    for segment,frame in data.items():
        validate_predictions(frame);p=frame.Probabilidad_default
        row=dict(Segmento=segment,N=len(frame),Minimo=float(p.min()),Maximo=float(p.max()),
            Promedio=float(p.mean()),Desv_estandar=float(p.std(ddof=0)),Tasa_default=float(frame.Target.mean()),
            Pct_clientes_prob_ge_050=float(p.ge(.5).mean()*100))
        row.update({f'P{q*100:g}':float(p.quantile(q)) for q in quantiles});rows.append(row)
    return pd.DataFrame(rows)


def internal_train_partition(train,cfg):
    """Partición para diagnóstico de umbrales; NO independiente del ajuste 5.0."""
    validate_predictions(train)
    if train.Target.value_counts().min()<2 or train.Target.nunique()!=2:
        raise ValueError('TRAIN insuficiente para partición interna estratificada')
    remainder_ids,validation_ids=train_test_split(train.index.to_numpy(),test_size=cfg['INTERNAL_VALIDATION_FRACTION'],
        stratify=train.Target.to_numpy(),random_state=cfg['RANDOM_STATE'])
    validation=train.loc[validation_ids].sort_index().copy()
    remainder=train.loc[remainder_ids].sort_index().copy()
    if set(validation.index)&set(remainder.index) or len(validation)+len(remainder)!=len(train):
        raise ValueError('Partición interna inválida')
    manifest=pd.DataFrame({'Particion':np.where(train.index.isin(validation.index),'TRAIN_VALIDACION_INTERNA','TRAIN_RESTO'),
        'Target':train.Target},index=train.index)
    return validation,remainder,manifest


def threshold_metrics(frame,threshold):
    validate_predictions(frame)
    if not np.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Umbral inválido')
    y=frame.Target.to_numpy(int);high=frame.Probabilidad_default.to_numpy()>=threshold
    tp=int(((y==1)&high).sum());fp=int(((y==0)&high).sum())
    tn=int(((y==0)&~high).sum());fn=int(((y==1)&~high).sum())
    notes=[]
    if tp+fp==0:notes.append('Precision indefinida: sin positivos predichos')
    if tp+fn==0:notes.append('Recall indefinido: sin defaults observados')
    if 2*tp+fp+fn==0:notes.append('F1 indefinido: sin positivos observados/predichos')
    return dict(Umbral=float(threshold),N=len(y),Precision=tp/(tp+fp) if tp+fp else np.nan,
        Recall=tp/(tp+fn) if tp+fn else np.nan,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
        Accuracy=(tp+tn)/len(y),TN=tn,FP=fp,FN=fn,TP=tp,Clientes_alto_riesgo=int(high.sum()),
        Pct_clientes_alto_riesgo=100*float(high.mean()),Advertencias='; '.join(notes))


def evaluate_grid(frame,grid=None):
    return pd.DataFrame([threshold_metrics(frame,t) for t in (threshold_grid() if grid is None else grid)])


def choose_candidates(internal_grid,cfg):
    """No acepta TEST/OOT. Criterios hipotéticos; no recomienda óptimo de negocio."""
    rows=[]
    def add(identifier,objective,subset,sort,ascending):
        if subset.empty:
            rows.append(dict(Candidato=identifier,Objetivo_hipotetico=objective,Umbral=np.nan,Estado='NO_FACTIBLE',
                Fuente='TRAIN_VALIDACION_INTERNA',Regla_desempate='Mayor umbral',Nota='Sin punto factible en grilla'))
            return
        best=subset.sort_values(sort,ascending=ascending,kind='stable').iloc[0]
        rows.append(dict(Candidato=identifier,Objetivo_hipotetico=objective,Umbral=float(best.Umbral),Estado='CONGELADO',
            Fuente='TRAIN_VALIDACION_INTERNA',Regla_desempate='Mayor umbral',Nota='Escenario diagnóstico; no óptimo operativo'))
    add('F1_INTERNO','Maximizar F1 interno; balance matemático sin costos de negocio',
        internal_grid.loc[internal_grid.F1.notna()],['F1','Umbral'],[False,False])
    add('RECALL_ESCENARIO',f"Capturar al menos {cfg['RECALL_SCENARIO']:.0%} de defaults con el mayor umbral factible",
        internal_grid.loc[internal_grid.Recall.ge(cfg['RECALL_SCENARIO'])],['Umbral'],[False])
    add('CAPACIDAD_ESCENARIO',f"Máxima captura con <= {cfg['MAX_HIGH_RISK_SCENARIO']:.0%} de clientes alto riesgo",
        internal_grid.loc[internal_grid.Pct_clientes_alto_riesgo.le(100*cfg['MAX_HIGH_RISK_SCENARIO'])&internal_grid.Recall.notna()],
        ['Recall','Umbral'],[False,False])
    rows.append(dict(Candidato='REFERENCIA_050',Objetivo_hipotetico='Referencia original 5.0; sin optimización',
        Umbral=cfg['REFERENCE_THRESHOLD'],Estado='CONGELADO',Fuente='CONFIGURACION_PREVIA',Regla_desempate='No aplica',Nota='No supone optimalidad'))
    return pd.DataFrame(rows)


def freeze_candidates(path,candidates,cfg,references,manifest_path):
    records=json.loads(candidates.to_json(orient='records',force_ascii=False,double_precision=15))
    artifact=dict(methodology_version='5.1',selection_source='TRAIN_VALIDACION_INTERNA',
        no_model_refit=True,limitation=LIMITATION,parameters=cfg,grid=threshold_grid().tolist(),
        candidates=records,references=references,partition_manifest=str(Path(manifest_path).resolve()),
        partition_manifest_sha256=sha256(manifest_path),
        business_note='Escenarios ilustrativos. No se declara umbral óptimo sin objetivo/costos de negocio.')
    Path(path).write_text(json.dumps(artifact,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return artifact


def load_candidates(path):
    artifact=json.loads(Path(path).read_text(encoding='utf-8'))
    if artifact['selection_source']!='TRAIN_VALIDACION_INTERNA' or not artifact['no_model_refit']:
        raise ValueError('Procedencia de selección inválida')
    for file,digest in artifact['references'].items():
        if sha256(file)!=digest:raise ValueError(f'Hash fuente alterado: {file}')
    if sha256(artifact['partition_manifest'])!=artifact['partition_manifest_sha256']:raise ValueError('Partición alterada')
    return artifact,pd.DataFrame(artifact['candidates'])


def evaluate_candidates(data,candidates):
    rows=[]
    for row in candidates.loc[candidates.Estado.eq('CONGELADO')].itertuples():
        for segment,frame in data.items():
            rows.append(dict(Candidato=row.Candidato,Segmento=segment,Objetivo_hipotetico=row.Objetivo_hipotetico,
                **threshold_metrics(frame,row.Umbral)))
    return pd.DataFrame(rows)


def risk_deciles(data,n_deciles=10):
    """Deciles descriptivos por muestra; iguales tamaños, ties por _row_id, sin target en ranking."""
    rows=[];assignments=[]
    for segment,frame in data.items():
        validate_predictions(frame)
        ordered=frame.assign(_orden_id=frame.index).sort_values(['Probabilidad_default','_orden_id'],ascending=[False,True])
        k=min(n_deciles,len(ordered));labels=np.floor(np.arange(len(ordered))*k/len(ordered)).astype(int)+1
        ordered=ordered.assign(Decil=labels)
        total_defaults=int(ordered.Target.sum());base=float(ordered.Target.mean());cum_n=0;cum_bad=0
        for decile,group in ordered.groupby('Decil',sort=True):
            n=len(group);bad=int(group.Target.sum());cum_n+=n;cum_bad+=bad
            previous=ordered.loc[ordered.Decil.eq(decile-1),'Probabilidad_default']
            ties=bool(len(previous) and group.Probabilidad_default.max()==previous.min())
            capture=cum_bad/total_defaults if total_defaults else np.nan
            rows.append(dict(Segmento=segment,Decil=int(decile),N=n,Defaults=bad,
                Tasa_default=bad/n,Probabilidad_promedio=float(group.Probabilidad_default.mean()),
                Probabilidad_min=float(group.Probabilidad_default.min()),Probabilidad_max=float(group.Probabilidad_default.max()),
                Pct_clientes=100*n/len(ordered),Captura_defaults_decil=bad/total_defaults if total_defaults else np.nan,
                Lift_decil=(bad/n)/base if base else np.nan,Clientes_acumulados=cum_n,
                Fraccion_clientes_acumulada=cum_n/len(ordered),Defaults_acumulados=cum_bad,
                Captura_defaults_acumulada=capture,Lift_acumulado=capture/(cum_n/len(ordered)),
                Empate_probabilidad_en_frontera=ties))
        assignments.append(ordered[['Decil']].assign(Segmento=segment).reset_index(names='_row_id'))
    return pd.DataFrame(rows),pd.concat(assignments,ignore_index=True)


def calibration_diagnostics(data,n_bins=10):
    rows=[];summary=[];base_train=float(data['TRAIN'].Target.mean())
    for segment,frame in data.items():
        validate_predictions(frame)
        bin_ids=pd.cut(frame.Probabilidad_default,bins=np.linspace(0,1,n_bins+1),labels=False,include_lowest=True,right=True)
        ece=0
        for i in range(n_bins):
            group=frame.loc[bin_ids.eq(i)];n=len(group)
            observed=float(group.Target.mean()) if n else np.nan;predicted=float(group.Probabilidad_default.mean()) if n else np.nan
            if n:ece+=n/len(frame)*abs(observed-predicted)
            rows.append(dict(Segmento=segment,Bin=i+1,Limite_inferior=i/n_bins,Limite_superior=(i+1)/n_bins,
                N=n,Defaults=int(group.Target.sum()),Tasa_default=observed,Probabilidad_promedio=predicted))
        brier=float(brier_score_loss(frame.Target,frame.Probabilidad_default))
        baseline=float(np.mean((frame.Target-base_train)**2));pred_mean=float(frame.Probabilidad_default.mean())
        summary.append(dict(Segmento=segment,N=len(frame),Brier=brier,Brier_constante_prevalencia_TRAIN=baseline,
            Brier_skill_vs_constante_TRAIN=1-brier/baseline if baseline else np.nan,
            Tasa_default=float(frame.Target.mean()),Probabilidad_promedio=pred_mean,
            Diferencia_predicha_observada=pred_mean-float(frame.Target.mean()),ECE_bins_fijos=ece,
            Observado_esperado=float(frame.Target.sum()/frame.Probabilidad_default.sum()) if pred_mean else np.nan))
    return pd.DataFrame(rows),pd.DataFrame(summary)


def plot_distribution(data):
    fig,axes=plt.subplots(1,3,figsize=(16,4),sharex=True)
    bins=np.linspace(0,1,51)
    for segment,ax in zip(SEGMENTS,axes):
        frame=data[segment]
        for target,label in [(0,'No default'),(1,'Default')]:
            values=frame.loc[frame.Target.eq(target),'Probabilidad_default']
            if len(values):ax.hist(values,bins=bins,weights=np.ones(len(values))/len(values),alpha=.55,label=label)
        ax.axvline(.5,color='black',ls='--',label='Referencia 0.50')
        ax.set(title=segment,xlabel='Probabilidad de default',ylabel='Fracción dentro de cada clase');ax.legend()
    fig.suptitle('Distribución de probabilidades congeladas por clase');fig.tight_layout();return fig


def plot_thresholds(grids,candidates):
    fig,axes=plt.subplots(1,3,figsize=(16,5),sharey=True)
    for segment,ax in zip(SEGMENTS,axes):
        group=grids.loc[grids.Segmento.eq(segment)]
        for column in ('Precision','Recall','F1'):ax.plot(group.Umbral,group[column],label=column)
        ax.plot(group.Umbral,group.Pct_clientes_alto_riesgo/100,ls='--',label='Fracción alto riesgo')
        for value in candidates.loc[candidates.Estado.eq('CONGELADO'),'Umbral'].unique():ax.axvline(value,color='gray',lw=.6,alpha=.5)
        ax.set(title=segment,xlabel='Umbral',ylabel='Proporción',ylim=(0,1));ax.legend(fontsize=8)
    fig.suptitle('Diagnóstico Precision / Recall vs umbral — sin selección con TEST/OOT');fig.tight_layout();return fig


def plot_capture(deciles):
    fig,ax=plt.subplots(figsize=(8,6))
    for segment in SEGMENTS:
        g=deciles.loc[deciles.Segmento.eq(segment)]
        ax.plot(np.r_[0,g.Fraccion_clientes_acumulada],np.r_[0,g.Captura_defaults_acumulada],'o-',label=segment)
    ax.plot([0,1],[0,1],'k--',label='Captura aleatoria');ax.set(xlabel='Fracción acumulada de clientes (mayor riesgo primero)',ylabel='Fracción acumulada de defaults',title='Captura acumulada por deciles descriptivos');ax.legend();fig.tight_layout();return fig


def plot_decile_default(deciles):
    fig,ax=plt.subplots(figsize=(9,5))
    for segment in SEGMENTS:
        g=deciles.loc[deciles.Segmento.eq(segment)];ax.plot(g.Decil,g.Tasa_default,'o-',label=segment)
    ax.set(xlabel='Decil: 1 = mayor riesgo',ylabel='Tasa de default observada',title='Defaults observados por decil de riesgo',xticks=range(1,11));ax.legend();fig.tight_layout();return fig


def plot_lift(deciles):
    fig,ax=plt.subplots(figsize=(9,5))
    for segment in SEGMENTS:
        g=deciles.loc[deciles.Segmento.eq(segment)];ax.plot(g.Decil,g.Lift_decil,'o-',label=segment)
    ax.axhline(1,color='gray',ls='--');ax.set(xlabel='Decil: 1 = mayor riesgo',ylabel='Lift del decil vs default global de la muestra',title='Lift por deciles',xticks=range(1,11));ax.legend();fig.tight_layout();return fig


def plot_calibration(table):
    fig,ax=plt.subplots(figsize=(7,6))
    for segment in SEGMENTS:
        g=table.loc[table.Segmento.eq(segment)&table.N.gt(0)]
        ax.plot(g.Probabilidad_promedio,g.Tasa_default,'o-',label=segment)
    ax.plot([0,1],[0,1],'k--');ax.set(xlabel='Probabilidad promedio',ylabel='Default observado',title='Calibración descriptiva — bins fijos [0,1]',xlim=(0,1),ylim=(0,1));ax.legend();fig.tight_layout();return fig


def export_diagnostics(root,tables,figures):
    root=Path(root)
    for folder in ('reportes','graficos','evidencias'):(root/folder).mkdir(parents=True,exist_ok=True)
    for name,table in tables.items():table.to_csv(root/f'reportes/{name}.csv',index=False)
    for name,fig in figures.items():fig.savefig(root/f'graficos/{name}.png',dpi=140,bbox_inches='tight')
