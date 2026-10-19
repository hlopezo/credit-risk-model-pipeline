"""Adaptación portable de rutas; algoritmos estadísticos heredados sin duplicar."""
from pathlib import Path
import json,os,shutil,sys,hashlib
REPO=Path(__file__).resolve().parents[1]
def prepare_runtime(repo=REPO):
    repo=Path(repo).resolve();cfg=json.loads((repo/'configs/pipeline.json').read_text(encoding='utf-8'))
    runtime=repo/cfg['paths']['runtime'];runtime.mkdir(parents=True,exist_ok=True)
    for folder in ('src','scripts','tests'):
        for source in (repo/folder).glob('*.py'):
            if source.name=='pipeline_runtime.py':continue
            target=runtime/source.name;content=source.read_bytes()
            if not target.exists() or target.read_bytes()!=content:target.write_bytes(content)
    for source in (repo/'notebooks').glob('*.ipynb'):
        target=runtime/source.name
        if not target.exists():shutil.copy2(source,target)
    raw=repo/cfg['paths']['raw'];legacy=runtime/'1.base'/raw.name;legacy.parent.mkdir(exist_ok=True)
    schema=json.loads((repo/'configs/data_schema.json').read_text())
    if hashlib.sha256(raw.read_bytes()).hexdigest()!=schema['sha256']:raise ValueError('Fuente autorizada alterada')
    if not legacy.exists():shutil.copy2(raw,legacy)
    if hashlib.sha256(legacy.read_bytes()).hexdigest()!=schema['sha256']:raise ValueError('Copia runtime de fuente original alterada')
    baseline=runtime/'reference_b2';baseline.mkdir(exist_ok=True)
    for source in (repo/'docs/reference/b2').rglob('*'):
        if source.is_file():
            target=baseline/source.relative_to(repo/'docs/reference/b2');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    evidence=runtime/'4.ANALIS_UV_BV_MV/BIVARIADO/evidencias';evidence.mkdir(parents=True,exist_ok=True)
    (evidence/'B3_base.json').write_text(json.dumps({'backup':'reference_b2','protected_hashes':{}},indent=2))
    # Comparación histórica B.2 no interviene en aprendizaje B.3.
    for name in ('mlflow_tracking.py','registrar_pipeline_mlflow.py'):
        file=runtime/name;file.write_text(file.read_text(encoding='utf-8').replace('http://127.0.0.1:5001',cfg['mlflow']['tracking_uri']),encoding='utf-8')
    # Los tres parámetros PSI que configure() no recibe se centralizan aquí.
    psi_file=runtime/'psi_temporal.py'
    psi_file.write_text(psi_file.read_text(encoding='utf-8')+'\n# Configuración portable de visualización/PSI.\n'+
        'N_BINS = '+repr(cfg['psi']['N_BINS'])+'\nAMBER = '+repr(cfg['psi']['AMBER'])+'\nRED = '+repr(cfg['psi']['RED'])+'\n',encoding='utf-8')
    tracking=runtime/'mlflow_tracking.py'
    tracking.write_text(tracking.read_text(encoding='utf-8').replace("EXPERIMENT='CREDIT_RISK_LOGISTIC_WOE'",'EXPERIMENT='+repr(cfg['mlflow']['experiment'])),encoding='utf-8')
    return runtime


def persist_psi_contract(env):
    """Contrato I/O para ejecutar el notebook 2 manualmente; sin recalcular filtros."""
    import hashlib,json
    root=Path('4.ANALIS_UV_BV_MV/PSI_TEMPORAL/evidencias');root.mkdir(parents=True,exist_ok=True)
    raw=Path('2.OUTPUT/1.1.0.TAB_SEGM_TRAIN_TEST_OOT.csv')
    record=dict(source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),v4_preserved=not env['variables_descartar_psi'],
        v3_columns=env['X_train_v3'].shape[1],v4_columns=env['X_train_v4'].shape[1],initial_predictors=env['X_train'].shape[1],
        configuration=env['configuracion_psi'],states=env['df_resumen_psi'].Estado_PSI.value_counts().to_dict(),
        execution='Notebook manual; contrato de entrada v4',previous_variability_excluded=env['variables_descartadas'],
        previous_missing_excluded=env['variables_descartadas_f3'])
    (root/'ejecucion.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    import importlib.util
    spec=importlib.util.spec_from_file_location('biv_bridge','3.0.0.bivariado.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.export_v4_from_psi()
