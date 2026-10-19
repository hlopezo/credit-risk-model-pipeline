"""Importa retrospectivamente fases 2-5 sin ejecutar notebooks ni ajustes."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import uuid
import numpy as np
import pandas as pd
import mlflow
from mlflow import MlflowClient
import mlflow_tracking as t

PSI=Path('4.ANALIS_UV_BV_MV/PSI_TEMPORAL')
BIV=Path('4.ANALIS_UV_BV_MV/BIVARIADO')
MULT=Path('4.ANALIS_UV_BV_MV/MULTIVARIADO')
MODEL=Path('5.MODELO_LOGISTICO')
NOTEBOOKS=['2.0.0.SCR_DES_ANLS_UNV_BIV_MULT.ipynb','3.0.0.SCR_DES_ANLS_BIVARIADO.ipynb',
    '4.0.0.SCR_DES_ANLS_Multivar.ipynb','5.0.0.SCR_DES_MODELO_LOGISTICO.ipynb']


def frame(path):return pd.read_csv(path,float_precision='round_trip')


def inspect_coherence():
    pointer=t.read_json(BIV/'entradas/LATEST.json');manifest_path=BIV/'entradas'/pointer['manifest']
    assert t.sha256(manifest_path)==pointer['manifest_sha256']
    manifest=t.read_json(manifest_path);binning=t.read_json(BIV/'binning_woe_train.json')
    for name,digest in manifest['files'].items():assert t.sha256(manifest_path.parent/name)==digest
    assert manifest['provenance']['psi_summary']==t.sha256(PSI/'reportes/df_resumen_psi.csv')
    assert manifest['provenance']['psi_report']==t.sha256(PSI/'reportes/df_reporte_univariado.csv')
    assert manifest['provenance']['psi_execution']==t.sha256(PSI/'evidencias/ejecucion.json')
    assert binning['input_manifest_sha256']==t.sha256(manifest_path) and binning['input_version']==manifest['version']
    selection=t.read_json(MULT/'seleccion_correlacion.json');configuration=t.read_json(MODEL/'configuracion_modelo.json')
    assert selection['woe_artifact_sha256']==t.sha256(BIV/'binning_woe_train.json')
    for name,digest in configuration['references'].items():assert t.sha256(name)==digest
    biv_summary=frame(BIV/'reportes/df_resumen_bivariado.csv');ranking=frame(MULT/'reportes/df_iv_final.csv')
    assert set(ranking.Variable)==set(binning['approved_variables'])==set(biv_summary.loc[biv_summary.Estado_bivariado.eq('APROBADA'),'Variable'])
    for r in biv_summary.itertuples():assert np.isclose(r.IV_total,binning['variables'][r.Variable]['iv_total'],rtol=1e-12,atol=1e-14)
    assert configuration['variables_received']==selection['retained_variables']
    coef=t.read_json(MODEL/'modelos/coeficientes.json');assert configuration['variables_final']==coef['variables']
    coefficients=frame(MODEL/'reportes/df_coeficientes_final.csv').set_index('Variable')
    np.testing.assert_array_equal(coefficients.loc[coef['variables'],'Coeficiente'],[coef['coefficients'][v] for v in coef['variables']])
    metrics=frame(MODEL/'reportes/df_metricas_modelo.csv');execution=t.read_json(MODEL/'evidencias/ejecucion.json')
    previous=pd.DataFrame(execution['metrics'])
    for column in ['AUC','Gini','KS','Recall','Precision','F1_score','Brier','Accuracy']:
        np.testing.assert_allclose(metrics[column],previous[column],rtol=1e-12,atol=1e-12)
    assert len(coef['variables'])==execution['number_final']==9
    assert metrics.Umbral.eq(configuration['parameters']['CLASSIFICATION_THRESHOLD']).all()
    # Solo evidencia de artefactos actuales. No se invoca lógica estadística previa.
    return dict(dataset_version=manifest['version'],manifest_sha256=t.sha256(manifest_path),
        binning_sha256=t.sha256(BIV/'binning_woe_train.json'),selection_sha256=t.sha256(MULT/'seleccion_correlacion.json'),
        coefficients_sha256=t.sha256(MODEL/'modelos/coeficientes.json'),
        model_sha256=t.sha256(MODEL/'modelos/modelo_logit.pkl'),metrics_sha256=t.sha256(MODEL/'reportes/df_metricas_modelo.csv'),
        psi_features=len(frame(PSI/'reportes/df_resumen_psi.csv')),bivariate_approved=len(binning['approved_variables']),
        multivariate_retained=len(selection['retained_variables']),final_predictors=len(coef['variables']))


def phase_definitions():
    psi_config=t.read_json(PSI/'reportes/parametros.json');psi_exec=t.read_json(PSI/'evidencias/ejecucion.json')
    univ=frame(PSI/'reportes/df_reporte_univariado.csv');psi=frame(PSI/'reportes/df_resumen_psi.csv')
    p1={**psi_config}
    for name,filter_name in [('variability_max_concentration_pct','Variabilidad'),('missing_max_pct','Nulos')]:
        values=univ.loc[univ.Filtro.eq(filter_name),'Umbral'].dropna().unique()
        assert len(values)==1;p1[name]=float(values[0])
    m1={'variables_initial':psi_exec['initial_predictors'],'variables_after_univariate':psi_exec['v3_columns'],
        'variables_final':psi_exec['v4_columns'],'psi_drop_candidates':int(psi.Candidata_descarte.sum())}
    m1.update({f'psi_state_{state.lower()}':int(n) for state,n in psi.Estado_PSI.value_counts().items()})
    binning=t.read_json(BIV/'binning_woe_train.json');summary=frame(BIV/'reportes/df_resumen_bivariado.csv')
    p2={k:binning['parameters'][k] for k in ('Q_VALUES','MIN_IV_TOTAL','MIN_BIN_PCT','MIN_RATE_GAP','ALPHA','POLICY_VERSION')}
    m2=dict(variables_initial=len(summary),variables_approved=len(binning['approved_variables']),variables_rejected=len(binning['rejected_variables']),
        variables_not_evaluable=len(binning['not_evaluable_variables']),selected_fusions_total=int(summary.N_fusiones.sum()))
    selection=t.read_json(MULT/'seleccion_correlacion.json')
    p3=dict(correlation_method=selection['correlation_method'],correlation_absolute_threshold=selection['threshold'],
        correlation_comparison=selection['comparison'],iv_tie_break=selection['iv_tie_break'])
    mult_exec=t.read_json(MULT/'evidencias/ejecucion.json')
    m3=dict(variables_initial=len(selection['iv_order']),variables_final=len(selection['retained_variables']),
        variables_eliminated=len(selection['eliminated_variables']),max_abs_final_correlation=mult_exec['max_abs_final_correlation'])
    config=t.read_json(MODEL/'configuracion_modelo.json');hist=frame(MODEL/'reportes/df_iteraciones_modelo.csv');last=hist.iloc[-1]
    p4={k:config['parameters'][k] for k in ('MAX_IV_MODEL','P_VALUE_THRESHOLD','SHAP_MIN_PCT','MAX_ITERACIONES','CLASSIFICATION_THRESHOLD','RANDOM_STATE','SHAP_SAMPLE_N','SHAP_BACKGROUND_N')}
    p4['woe_convention']=config['convention']['formula'];p4['expected_sign']=config['convention']['expected_sign']
    m4=dict(variables_initial=len(config['variables_received']),variables_final=len(config['variables_final']),
        variables_excluded_iv=int(frame(MODEL/'reportes/variables_excluidas_iv.csv').shape[0]),
        aic=last.AIC,bic=last.BIC,log_likelihood=last.Log_Likelihood,pseudo_r2=last.Pseudo_R2,converged=int(last.Estado_convergencia))
    for criterion in ('SIGNO','P_VALUE','SHAP'):m4[f'eliminated_{criterion.lower()}']=int(hist.Criterio_eliminacion.eq(criterion).sum())
    mapping={'AUC':'auc','Gini':'gini','KS':'ks','Recall':'recall','Precision':'precision','F1_score':'f1','Brier':'brier','Accuracy':'accuracy',
        'Tasa_default_observada':'default_rate','Probabilidad_promedio':'mean_probability'}
    for _,row in frame(MODEL/'reportes/df_metricas_modelo.csv').iterrows():
        for original,key in mapping.items():m4[f'{row.Segmento.lower()}_{key}']=row[original]
    def reports(root,excluded=()):return [p for p in sorted((root/'reportes').glob('*.csv')) if p.name not in excluded]
    def graphics(root,folders):return [p for folder in folders for p in sorted((root/folder).glob('*.png'))]
    defs=[
        dict(root=PSI,params=p1,metrics=m1,methodology='PSI_TEMPORAL_MENSUAL',files=reports(PSI)+[PSI/'reportes/bins.json',PSI/'reportes/parametros.json']+graphics(PSI,['subplots','heatmaps'])),
        dict(root=BIV,params=p2,metrics=m2,methodology='B.3',files=reports(BIV)+[BIV/'binning_woe_train.json']+graphics(BIV,['subplots','graficos_individuales','estabilidad'])),
        dict(root=MULT,params=p3,metrics=m3,methodology='4.0',files=reports(MULT)+[MULT/'seleccion_correlacion.json']+graphics(MULT,['graficos'])),
        dict(root=MODEL,params=p4,metrics=m4,methodology='5.0',files=reports(MODEL,['shap_valores_train.csv'])+[MODEL/'reportes/summary_statsmodels.txt',MODEL/'configuracion_modelo.json']+graphics(MODEL,['graficos']))]
    return defs


@t.local_tracking_session()
def import_pipeline(new_import=False):
    coherence=inspect_coherence();definitions=phase_definitions()
    evidence=t.LOCAL/'evidencias';evidence.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(json.dumps(coherence,sort_keys=True).encode()).hexdigest()
    report_path=evidence/'importacion.json';experiment=t.configure()
    client=MlflowClient()
    if report_path.exists() and not new_import:
        previous=t.read_json(report_path)
        if previous['pipeline_fingerprint']==fingerprint and all(client.get_run(r).info.status=='FINISHED' for r in previous['run_ids']):
            print('Importación existente reutilizada, sin duplicar runs.');return previous
    pipeline_id=fingerprint[:16]+('_'+uuid.uuid4().hex[:8] if new_import else '')
    # Copia de hashes locales solo en evidencia local, no publica matrices ni predicciones.
    protected={}
    for folder in ('4.ANALIS_UV_BV_MV','5.MODELO_LOGISTICO','5.1.EVALUACION_UMBRALES'):
        protected.update({p.as_posix():t.sha256(p) for p in Path(folder).rglob('*') if p.is_file()})
    protected.update({p.as_posix():t.sha256(p) for p in Path('.').glob('*.ipynb')})
    t.write_json(evidence/'hashes_previos.json',protected)
    run_ids=[];manifest_counts=[];model_info=None
    for i,definition in enumerate(definitions):
        name=t.PHASES[i]
        prior=client.search_runs([experiment],filter_string=f"tags.pipeline_id = '{pipeline_id}' and tags.phase = '{i+2}.0'")
        finished=[r for r in prior if r.info.status=='FINISHED']
        resume_id=finished[0].info.run_id if len(finished)==1 else None
        for failed in prior:
            if failed.info.status=='FAILED':
                client.set_tag(failed.info.run_id,'superseded_import_attempt','true')
                client.delete_run(failed.info.run_id)  # Borrado lógico reversible; se preserva auditoría SQLite.
        tags=dict(phase=str(i+2)+'.0',methodology=definition['methodology'],model_type='logistic_woe',
            dataset_version=coherence['dataset_version'],execution_type='retrospective_import',
            source_notebook=NOTEBOOKS[i],pipeline_id=pipeline_id,pipeline_fingerprint=fingerprint,
            original_execution_date_status='not_recorded_in_structured_evidence')
        if run_ids:tags['upstream_run_id']=run_ids[-1]
        if i==1:tags['original_artifact_created_at']=t.read_json(BIV/'binning_woe_train.json')['created_at']
        if i==3:
            text=(MODEL/'reportes/summary_statsmodels.txt').read_text(encoding='utf-8')
            date=re.search(r'Date:\s+(.+?)\s+Pseudo',text)
            if date:tags['original_report_date']=date.group(1);tags['original_execution_date_status']='date_reported_by_statsmodels_timezone_not_recorded'
        stage=t.LOCAL/'staging'/pipeline_id/name
        with t.phase_run(experiment,name,tags,definition['params'],definition['metrics'],run_id=resume_id) as run:
            run_ids.append(run.info.run_id)
            if i==0:
                from segmentation_validation import require_generated_partition
                root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'configs/pipeline.json').exists())
                phase1=require_generated_partition(root)
                mlflow.log_dict(phase1,'provenance/phase1_partition_validation.json')
                mlflow.set_tag('phase1_source_notebook','1.0.0.SCR_DES_MOD_RL_SEGMT.ipynb')
                mlflow.set_tag('original_unsegmented_sha256',phase1['input_raw_sha256'])
            if i==3:
                cached=evidence/f'model_{pipeline_id}.json'
                if cached.exists():model_info=t.read_json(cached)
                else:
                    model_info=t.register_frozen_model(MODEL/'modelos/coeficientes.json',t.LOCAL/'model_build'/pipeline_id)
                    t.write_json(cached,model_info)
            manifest=t.log_artifacts(stage,definition['files']);manifest_counts.append(len(manifest))
            mlflow.log_dict(coherence,'provenance/pipeline_coherence.json')
            inventory=[]
            modules={0:['psi_temporal.py','ejecutar_psi_notebook.py','informe_psi.py'],1:['3.0.0.bivariado.py','3.0.1.ejecutar_bivariado.py'],
                2:['4.0.0.multivariado.py','4.0.1.ejecutar_multivariado.py'],3:['5.0.0.modelo_logistico.py','5.0.1.ejecutar_modelo_logistico.py']}[i]
            for file in [NOTEBOOKS[i]]+modules:
                inventory.append(dict(source_relative=file,sha256=t.sha256(file),content_uploaded=False))
            mlflow.log_dict(dict(notebooks_modules=inventory,output_root=definition['root'].as_posix(),
                excluded=['matrices','entradas','predicciones','individual SHAP','notebook outputs','original statsmodels pickle'],
                upstream_run_id=run_ids[i-1] if i else None),'provenance/source_inventory.json')
            mlflow.log_dict(dict(import_type='retrospective_import',original_execution_not_performed_by_mlflow=True,
                original_dates_policy='Only persisted structured creation dates / original report date; no guessed timestamps'),
                'provenance/import_declaration.json')
    for run_id in run_ids:
        client.set_tag(run_id,'pipeline_run_ids',json.dumps(run_ids))
        client.set_tag(run_id,'final_model_run_id',run_ids[-1])
    scan=t.scan_published_tree(t.LOCAL/'artifacts')
    for path,digest in protected.items():assert t.sha256(path)==digest,path
    import statsmodels.api as sm
    original=sm.load(MODEL/'modelos/modelo_logit.pkl')
    example=t.synthetic_example(t.read_json(MODEL/'modelos/coeficientes.json'))
    loaded=mlflow.pyfunc.load_model(f"models:/{model_info['registered_name']}/{model_info['registered_version']}")
    np.testing.assert_allclose(loaded.predict(example).probabilidad_default,original.predict(sm.add_constant(example,has_constant='add')),rtol=1e-12,atol=1e-12)
    report=dict(experiment_name=t.EXPERIMENT,experiment_id=experiment,tracking_uri=t.TRACKING_URI,
        backend='sqlite:///MLFLOW_LOCAL/mlflow.db',artifact_root='MLFLOW_LOCAL/artifacts',pipeline_id=pipeline_id,
        pipeline_fingerprint=fingerprint,run_ids=run_ids,run_names=list(t.PHASES),coherence=coherence,
        artifacts_per_phase=manifest_counts,model=model_info,privacy_scan=scan,previous_files_unchanged=len(protected),
        synthetic_predictions_match_original_model=True,execution_type='retrospective_import',training_performed=False)
    t.write_json(report_path,report)
    print(json.dumps(report,indent=2,ensure_ascii=False));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--new-import',action='store_true')
    import_pipeline(parser.parse_args().new_import)
