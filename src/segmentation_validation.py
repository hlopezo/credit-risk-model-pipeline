"""Auditoría de la salida original: valida, no crea ni modifica particiones."""
from pathlib import Path
import hashlib,json
import pandas as pd

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def values_hash(values):return hashlib.sha256(json.dumps(list(map(int,values)),separators=(',',':')).encode()).hexdigest()
def rows_hash(frame):return hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes()).hexdigest()
def validate_generated_partition(repo):
    repo=Path(repo);cfg=json.loads((repo/'configs/pipeline.json').read_text());schema=json.loads((repo/'configs/data_schema.json').read_text())
    ref=json.loads((repo/'configs/partition_reference.json').read_text());runtime=repo/cfg['paths']['runtime'];file=runtime/cfg['paths']['segmented']
    raw=repo/schema['file']
    if sha256(raw)!=schema['sha256']:raise ValueError('Base original alterada')
    if not file.exists():raise ValueError('Falta ejecutar notebook 1.0; no hay partición generada')
    actual=pd.read_csv(file)
    if sha256(file)!=ref['sha256']:raise ValueError('CSV segmentado diferente al original: detener antes del modelamiento')
    if actual.columns.tolist()!=ref['columns'] or {c:str(d) for c,d in actual.dtypes.items()}!=ref['dtypes']:
        raise ValueError('Columnas/tipos de la partición no coinciden')
    if len(actual)!=ref['records'] or actual.SEGMENTO.isna().any() or set(actual.SEGMENTO)!=set(ref['segments']):raise ValueError('Filas/segmentos inválidos')
    rows=[]
    for segment,expected in ref['segments'].items():
        part=actual.loc[actual.SEGMENTO.eq(segment)]
        observed=dict(records=len(part),clients=int(part.cliente_id.nunique()),target_counts={str(k):int(v) for k,v in part.indicador_default.value_counts().items()},periods=sorted(map(int,part.periodo_observacion.unique())),index_sha256=values_hash(part.index),identifiers_sha256=values_hash(part.cliente_id),target_sha256=values_hash(part.indicador_default),rows_with_indices_sha256=rows_hash(part))
        if observed!=expected:raise ValueError(f'{segment}: membresía, índices, IDs, target, períodos o conteos no coinciden')
        rows.append(dict(Segmento=segment,Registros=len(part),Clientes=observed['clients'],Defaults=int(part.indicador_default.sum()),Tasa_default=float(part.indicador_default.mean()),Periodos=observed['periods'],Coincide_exactamente=True))
    result=dict(phase1_executed=True,phase1_notebook='1.0.0.SCR_DES_MOD_RL_SEGMT.ipynb',input_raw_sha256=schema['sha256'],segmented_file_sha256=sha256(file),original_segmented_sha256=ref['sha256'],columns_types_match=True,rows_indices_identifiers_targets_match=True,segment_comparison=rows,source='executed notebook 1.0; original input without SEGMENTO',segmentation=cfg['segmentation'])
    evidence=runtime/'1.SEGMENTACION/evidencias';evidence.mkdir(parents=True,exist_ok=True)
    (evidence/'particiones_verificadas.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result

def require_generated_partition(repo):
    repo=Path(repo);cfg=json.loads((repo/'configs/pipeline.json').read_text());runtime=repo/cfg['paths']['runtime']
    evidence=runtime/'1.SEGMENTACION/evidencias/particiones_verificadas.json'
    if not evidence.exists():raise ValueError('Ejecute primero fase 1.0: prohibido usar particiones precalculadas sin evidencia de generación')
    result=json.loads(evidence.read_text())
    if result.get('phase1_executed') is not True:raise ValueError('No consta ejecución de fase 1.0')
    if sha256(runtime/cfg['paths']['segmented'])!=result['segmented_file_sha256']:raise ValueError('Partición modificada tras fase 1.0')
    if sha256(repo/cfg['paths']['raw'])!=result['input_raw_sha256']:raise ValueError('Fuente modificada tras fase 1.0')
    return result
