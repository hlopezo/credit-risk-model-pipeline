"""Tracking local 5.2. Solo agregados aprobados; sin autologging ni entrenamiento."""
import os
os.environ['MLFLOW_DISABLE_TELEMETRY']='true'
os.environ['MLFLOW_DISABLE_AGENT_HINT']='1'
os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING']='false'
from pathlib import Path
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import re
import sys
import tempfile
import uuid
import yaml
import numpy as np
import pandas as pd
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

EXPERIMENT='CREDIT_RISK_LOGISTIC_WOE'
TRACKING_URI='http://127.0.0.1:5001'
LOCAL=Path('MLFLOW_LOCAL')
PHASES=('01_UNIVARIADO_PSI','02_BIVARIADO_WOE','03_MULTIVARIADO_CORRELACION','04_MODELO_LOGISTICO_FINAL')
DENIED_COLUMNS={'_row_id','cliente_id','dni','documento','nro_documento','target','indicador_default',
    'probabilidad_default','prediccion','row_ids','sample_indices','background_indices'}
SENSITIVE_KEYS={'password','secret','token','credential','api_key','username'}
ABSOLUTE_PATH=re.compile(r'(?i)(?<![a-z0-9_])(?:[a-z]:[\\/]|/Users/|/home/|/mnt/)')


@contextmanager
def local_tracking_session():
    """Compatibilidad Python 3.14/Windows: temporales con ACL heredada del workspace.

    Alcance limitado a esta sesión; no modifica ACL existentes ni archivos de Python.
    """
    if os.name!='nt':
        yield;return
    original=tempfile.mkdtemp
    root=(LOCAL/'tmp').resolve();root.mkdir(parents=True,exist_ok=True)
    def inherited_temp(suffix=None,prefix=None,dir=None):
        parent=Path(dir).resolve() if dir else root
        if not parent.is_relative_to(Path.cwd().resolve()):parent=root
        path=parent/((prefix or 'tmp')+uuid.uuid4().hex+(suffix or ''))
        path.mkdir(parents=True)
        return str(path)
    tempfile.mkdtemp=inherited_temp
    try:yield
    finally:tempfile.mkdtemp=original


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def scrub(value):
    if isinstance(value,dict):
        result={}
        for k,v in value.items():
            metadata_target=str(k).lower()=='target' and v=='indicador_default'
            if str(k).lower() in SENSITIVE_KEYS or (str(k).lower() in DENIED_COLUMNS and not metadata_target):raise ValueError(f'Clave sensible: {k}')
            key=str(k)
            if ABSOLUTE_PATH.search(key):
                # Se conservan referencias relativas al workspace, sin prefijo personal.
                path=Path(key)
                try:key=path.resolve().relative_to(Path.cwd().resolve()).as_posix()
                except ValueError:key='local_reference_'+hashlib.sha256(key.encode()).hexdigest()[:12]
            result[key]=scrub(v)
        return result
    if isinstance(value,list):return [scrub(v) for v in value]
    if isinstance(value,str):
        if ABSOLUTE_PATH.search(value):
            try:return Path(value).resolve().relative_to(Path.cwd().resolve()).as_posix()
            except ValueError:return '[local_path_redacted]'
        return value
    if isinstance(value,float) and not np.isfinite(value):return str(value)
    return value


def validate_aggregate_csv(path):
    columns=pd.read_csv(path,nrows=0).columns.tolist()
    denied=[c for c in columns if c.casefold() in DENIED_COLUMNS]
    if denied:raise ValueError(f'CSV con datos individuales/identificadores: {denied}')
    # Identificación por grano: cada fila debe representar variable, período o resumen de muestra/modelo.
    if not any(c in columns for c in ('Variable','Segmento','Periodo','Iteracion','Comparacion','Estado_PSI','Unnamed: 0')):
        raise ValueError('CSV sin esquema agregado aprobado')


def configure(tracking_uri=TRACKING_URI,experiment=EXPERIMENT):
    if not re.fullmatch(r'http://(?:127\.0\.0\.1|localhost):\d+',tracking_uri):
        raise ValueError('Esta integración autoriza únicamente tracking HTTP local')
    mlflow.set_tracking_uri(tracking_uri);mlflow.autolog(disable=True)
    existing=mlflow.get_experiment_by_name(experiment)
    if existing is None:
        identifier=mlflow.create_experiment(experiment,artifact_location=f'mlflow-artifacts:/{experiment}')
    else:identifier=existing.experiment_id
    return identifier


def log_parameters(parameters):
    for key,value in scrub(parameters).items():
        if any(word in key.casefold() for word in SENSITIVE_KEYS):raise ValueError('Parámetro sensible')
        if isinstance(value,(list,dict)):value=json.dumps(value,ensure_ascii=False,allow_nan=False)
        mlflow.log_param(key,value)


def log_metrics(metrics):
    missing=[]
    for key,value in metrics.items():
        if value is None or not np.isfinite(float(value)):missing.append(key);continue
        mlflow.log_metric(key,float(value))
    if missing:mlflow.log_dict({'undefined_metrics':missing},'provenance/undefined_metrics.json')


def log_versions():
    versions={name:importlib.metadata.version(name) for name in ('mlflow','numpy','pandas','statsmodels','scikit-learn','shap')}
    versions['python']=sys.version.split()[0]
    mlflow.log_dict(versions,'provenance/runtime_import_versions.json')
    mlflow.set_tag('runtime_versions_role','import_environment_not_original_execution')


@contextmanager
def phase_run(experiment_id,name,tags,parameters=None,metrics=None,run_id=None):
    """Hook futuro: el llamador calcula; esta capa solo registra resultados entregados."""
    source_name=('registrar_pipeline_mlflow.py' if tags.get('execution_type')=='retrospective_import'
        else tags.get('source_notebook','future_execution'))
    safe_tags=scrub({**tags,'mlflow.user':'local_operator','mlflow.source.name':source_name})
    with mlflow.start_run(experiment_id=experiment_id,run_name=name,tags=safe_tags,run_id=run_id) as run:
        if parameters:log_parameters(parameters)
        if metrics:log_metrics(metrics)
        log_versions()
        yield run


def stage_file(source,destination):
    """Copia depurada, jamás directorios de datos ni pickle statsmodels original."""
    source,destination=Path(source),Path(destination)
    if source.suffix.lower() not in ('.csv','.json','.png','.txt'):raise ValueError('Formato no aprobado')
    if any(part in ('matrices','predicciones','entradas') for part in source.parts):raise ValueError('No publicar matrices/predicciones/entradas')
    if source.name in ('shap_valores_train.csv','manifest_indices.json'):raise ValueError('Archivo individual no autorizado')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if source.suffix=='.csv':
        validate_aggregate_csv(source)
        frame=pd.read_csv(source,float_precision='round_trip')
        for col in frame.select_dtypes(include=['str','object']).columns:frame[col]=frame[col].map(lambda x:scrub(x) if isinstance(x,str) else x)
        frame.to_csv(destination,index=False)
    elif source.suffix=='.json':write_json(destination,scrub(read_json(source)))
    elif source.suffix=='.txt':
        text=source.read_text(encoding='utf-8')
        if ABSOLUTE_PATH.search(text):raise ValueError('Texto con ruta personal')
        destination.write_text(text,encoding='utf-8')
    else:destination.write_bytes(source.read_bytes())
    return dict(source_relative=source.as_posix(),source_sha256=sha256(source),
        artifact_relative=destination.name,published_sha256=sha256(destination),bytes=destination.stat().st_size)


def log_artifacts(stage,entries):
    """La importación llama tras construir una lista explícita de reportes permitidos."""
    manifest=[]
    for entry in entries:
        source=Path(entry)
        relative=(Path('graficos')/source.parent.name/source.name if source.suffix=='.png' else Path('reportes')/source.name)
        target=Path(stage)/relative
        record=stage_file(source,target);record['artifact_relative']=relative.as_posix();manifest.append(record)
    write_json(Path(stage)/'provenance/artifact_manifest.json',manifest)
    mlflow.log_artifacts(str(stage))
    return manifest


MODEL_CODE='''import json
import numpy as np
import pandas as pd
import mlflow

class FrozenWoELogit(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["coefficients"], encoding="utf-8") as stream:
            self.spec = json.load(stream)
    def predict(self, context, model_input, params=None):
        columns = self.spec["variables"]
        if model_input.columns.tolist() != columns:
            raise ValueError("Se requieren los nueve predictores WoE congelados en el orden de la firma")
        values = model_input.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("WoE no finito")
        weights = np.array([self.spec["coefficients"][c] for c in columns])
        logit = self.spec["intercept"] + values @ weights
        probabilities = np.empty_like(logit)
        positive = logit >= 0
        probabilities[positive] = 1 / (1 + np.exp(-logit[positive]))
        exponential = np.exp(logit[~positive])
        probabilities[~positive] = exponential / (1 + exponential)
        return pd.DataFrame({"probabilidad_default": probabilities}, index=model_input.index)

mlflow.models.set_model(FrozenWoELogit())
'''


def synthetic_example(coefficients):
    features=coefficients['variables'];n=len(features)
    return pd.DataFrame(np.vstack([np.zeros(n),np.linspace(-.5,.5,n),np.linspace(.75,-.75,n)]),columns=features,dtype=float)


def register_frozen_model(coefficients_path,stage):
    """Equivalente de inferencia; no incluye resultado statsmodels con endog/exog."""
    stage=Path(stage);stage.mkdir(parents=True,exist_ok=True)
    coefficients=read_json(coefficients_path)
    safe_path=stage/'coefficients.json';write_json(safe_path,coefficients)
    code_path=stage/'model_code.py';code_path.write_text(MODEL_CODE,encoding='utf-8')
    example=synthetic_example(coefficients)
    weights=np.array([coefficients['coefficients'][c] for c in coefficients['variables']])
    expected=1/(1+np.exp(-(coefficients['intercept']+example.to_numpy()@weights)))
    signature=infer_signature(example,pd.DataFrame({'probabilidad_default':expected}))
    saved=stage/('safe_model_'+uuid.uuid4().hex[:8])
    mlflow.pyfunc.save_model(path=str(saved),python_model=str(code_path),
        artifacts={'coefficients':str(safe_path)},signature=signature,input_example=example,
        pip_requirements=[f'{p}=={importlib.metadata.version(p)}' for p in ('mlflow','numpy','pandas')],
        metadata={'model_kind':'frozen_statsmodels_logit_inference_equivalent','training_data_included':False,'input_space':'WoE B.3 congelado'})
    # MLflow guarda por defecto model_code_path absoluto: depurar ANTES de publicar.
    model_file=saved/'MLmodel';document=yaml.safe_load(model_file.read_text(encoding='utf-8'))
    document['flavors']['python_function']['model_code_path']='model_code.py'
    client=MlflowClient();active=mlflow.active_run()
    logged=client.create_logged_model(active.info.experiment_id,name='logit_woe',source_run_id=active.info.run_id,model_type='logistic_woe')
    document.update(model_id=logged.model_id,model_uuid=logged.model_id,run_id=active.info.run_id)
    model_file.write_text(yaml.safe_dump(document,sort_keys=False,allow_unicode=True),encoding='utf-8')
    scan_published_tree(saved)
    client.log_model_artifacts(logged.model_id,str(saved));client.finalize_logged_model(logged.model_id,'READY')
    model_uri=f'models:/{logged.model_id}'
    loaded=mlflow.pyfunc.load_model(model_uri)
    np.testing.assert_allclose(loaded.predict(example).probabilidad_default,expected,rtol=1e-12,atol=1e-12)
    registered=mlflow.register_model(model_uri,'CREDIT_RISK_LOGISTIC_WOE_FINAL')
    client.set_registered_model_tag('CREDIT_RISK_LOGISTIC_WOE_FINAL','input_space','frozen_woe')
    client.set_model_version_tag(registered.name,registered.version,'execution_type',
        active.data.tags.get('execution_type','native_execution'))
    return dict(model_uri=model_uri,registered_name=registered.name,registered_version=registered.version,
        input_columns=coefficients['variables'],synthetic_predictions=expected.tolist(),signature_verified=True)


def scan_published_tree(root):
    """Auditoría de texto/cabeceras; no interpreta números agregados como DNI."""
    findings=[];root=Path(root)
    for p in root.rglob('*'):
        if not p.is_file():continue
        if p.suffix.lower() in ('.pkl','.pickle','.parquet','.ipynb'):findings.append(str(p.relative_to(root)))
        if p.suffix.lower() in ('.csv','.json','.yaml','.yml','.txt','.py') or p.name=='MLmodel':
            text=p.read_text(encoding='utf-8',errors='replace')
            if ABSOLUTE_PATH.search(text):findings.append('personal_path:'+str(p.relative_to(root)))
            if p.suffix=='.csv':
                columns=pd.read_csv(p,nrows=0).columns
                if any(c.casefold() in DENIED_COLUMNS for c in columns):findings.append('individual_columns:'+str(p.relative_to(root)))
    if findings:raise ValueError(f'Artefactos no aptos: {findings}')
    return dict(files_scanned=sum(p.is_file() for p in root.rglob('*')),sensitive_findings=0)
