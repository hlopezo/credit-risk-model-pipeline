"""Selección 4.0 por Pearson WoE TRAIN con prioridad IV; no estima WoE ni modelos."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEGMENTS = ('TRAIN', 'TEST', 'OOT')
ROOT = Path('4.ANALIS_UV_BV_MV/MULTIVARIADO')
BIV_ROOT = Path('4.ANALIS_UV_BV_MV/BIVARIADO')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_bivariado():
    spec = importlib.util.spec_from_file_location('bivariado_mult', '3.0.0.bivariado.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_inputs(bundle, expected_features=29):
    rows = []
    columns = bundle['X']['TRAIN'].columns
    for segment in SEGMENTS:
        X, y, meta = (bundle[k][segment] for k in ('X', 'y', 'metadata'))
        if (X.columns.has_duplicates or X.index.has_duplicates or len(X) == 0
                or len(columns) != expected_features or not X.columns.equals(columns)):
            raise ValueError(f'{segment}: columnas/registros originales incompatibles')
        if not X.index.equals(y.index) or not X.index.equals(meta.index):
            raise ValueError(f'{segment}: índices X/target/metadata desalineados')
        if y.isna().any() or not y.isin([0, 1]).all() or not meta.SEGMENTO.eq(segment).all():
            raise ValueError(f'{segment}: target o segmento inválido')
        pd.testing.assert_series_equal(y, meta.indicador_default, check_exact=True)
        rows.append(dict(Segmento=segment, Registros=len(X), Variables=X.shape[1], Estado='VALIDADO'))
    return pd.DataFrame(rows)


def frozen_transform(bundle, artifact, biv, exported_dir=None):
    """Delega exclusivamente en transform_woe de 3.0; compara CSV si existen."""
    if artifact.get('methodology_version') != 'B.3':
        raise ValueError('Se requiere el artefacto congelado B.3')
    matrices, diagnostics = {}, []
    for segment in SEGMENTS:
        result, diag = biv.transform_woe(bundle['X'][segment], artifact, approved_only=True)
        if (result.columns.tolist() != artifact['approved_variables']
                or not result.index.equals(bundle['X'][segment].index)
                or not np.isfinite(result.to_numpy(float)).all()):
            raise ValueError(f'{segment}: transformación WoE inválida')
        if exported_dir is not None:
            file = Path(exported_dir)/f'X_{segment.lower()}_woe.csv'
            if file.exists():
                previous = pd.read_csv(file, index_col=0, float_precision='round_trip')
                previous.index.name = result.index.name
                pd.testing.assert_frame_equal(result, previous, check_exact=True)
        matrices[segment] = result
        diagnostics.append(diag.assign(Segmento=segment))
    return matrices, pd.concat(diagnostics, ignore_index=True)


def rank_iv(summary, artifact, columns):
    required = {'Variable', 'Estado_bivariado', 'IV_total', 'N_bins_efectivos'}
    if not required.issubset(summary.columns) or summary.Variable.duplicated().any():
        raise ValueError('Resumen bivariado: esquema o variables duplicadas inválidos')
    approved = summary.loc[summary.Estado_bivariado.eq('APROBADA')].copy()
    if (set(approved.Variable) != set(columns) or set(columns) != set(artifact['approved_variables'])
            or len(columns) != len(set(columns))):
        raise ValueError('Diferencia entre aprobadas del resumen, JSON y columnas WoE; detener selección')
    for row in approved.itertuples():
        model = artifact['variables'][row.Variable]
        if (model['state'] != 'APROBADA' or not np.isfinite(row.IV_total)
                or row.IV_total < 0 or not np.isclose(row.IV_total, model['iv_total'], rtol=1e-12, atol=1e-14)
                or row.N_bins_efectivos != model['effective_bins']):
            raise ValueError(f'IV/bins final no corresponde al JSON: {row.Variable}')
    result = approved[['Variable', 'IV_total']].rename(columns={'IV_total': 'IV_final'})
    result = result.sort_values(['IV_final', 'Variable'], ascending=[False, True], kind='stable').reset_index(drop=True)
    result['Ranking_IV'] = np.arange(1, len(result)+1)
    return result


def train_correlation(X_train, order):
    """Solo acepta la matriz TRAIN; NaN/constantes detienen sin imputar correlación."""
    if X_train.columns.has_duplicates or set(order) != set(X_train.columns) or len(order) != len(set(order)):
        raise ValueError('Orden y columnas TRAIN incompatibles')
    X = X_train.loc[:, order]
    if len(X) < 2 or not np.isfinite(X.to_numpy(float)).all():
        raise ValueError('TRAIN insuficiente o WoE no finito')
    constants = X.columns[X.nunique(dropna=False).le(1)].tolist()
    if constants:
        raise ValueError(f'WoE constante; correlación indefinida. Resolver antes de seleccionar: {constants}')
    corr = X.corr(method='pearson')
    validate_correlation(corr, order)
    return corr


def validate_correlation(corr, order):
    values = corr.to_numpy(float)
    if (corr.index.tolist() != list(order) or corr.columns.tolist() != list(order)
            or not np.isfinite(values).all() or not np.allclose(values, values.T)
            or not np.allclose(np.diag(values), 1) or (np.abs(values)>1+1e-12).any()):
        raise ValueError('Correlación indefinida, asimétrica o incompatible; NaN no equivale a cero')


def select_sequential(iv, corr, threshold=0.60):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Umbral debe estar entre 0 y 1')
    expected = iv.sort_values(['IV_final', 'Variable'], ascending=[False, True]).Variable.tolist()
    if iv.Variable.tolist() != expected or iv.Variable.duplicated().any() or not np.isfinite(iv.IV_final).all():
        raise ValueError('Ranking IV no ordenado o inválido')
    validate_correlation(corr, expected)
    retained, decisions, details = [], [], []
    scores = iv.set_index('Variable').IV_final
    for row in iv.itertuples(index=False):
        hits = [(ref, float(corr.loc[row.Variable, ref])) for ref in retained
                if abs(corr.loc[row.Variable, ref]) > threshold]
        # Mayor |r|; empate por orden IV previamente conservado.
        hits.sort(key=lambda item: -abs(item[1]))
        ref, coefficient = hits[0] if hits else (None, None)
        reason = (f'|r|={abs(coefficient):.8f} > {threshold}; prioridad IV de {ref}' if hits
                  else 'Sin correlación absoluta superior al umbral con variables previamente conservadas')
        decisions.append(dict(Variable=row.Variable, Ranking_IV=int(row.Ranking_IV), IV_final=float(row.IV_final),
            Estado_correlacion='ELIMINADA_CORRELACION' if hits else 'CONSERVADA',
            Variable_retenida_de_referencia=ref, IV_variable_retenida=float(scores[ref]) if hits else None,
            Correlacion_Pearson=coefficient, Correlacion_absoluta=abs(coefficient) if hits else None,
            Umbral=float(threshold), Motivo_decision=reason))
        for reference, value in hits:
            details.append(dict(Variable=row.Variable, Variable_retenida=reference, IV_final=float(row.IV_final),
                IV_variable_retenida=float(scores[reference]), Correlacion_Pearson=value,
                Correlacion_absoluta=abs(value), Umbral=float(threshold)))
        if not hits:
            retained.append(row.Variable)
    detail_columns=['Variable','Variable_retenida','IV_final','IV_variable_retenida','Correlacion_Pearson','Correlacion_absoluta','Umbral']
    return retained, pd.DataFrame(decisions), pd.DataFrame(details, columns=detail_columns)


def apply_selection(matrices, retained):
    if not retained or len(retained) != len(set(retained)):
        raise ValueError('Selección vacía o duplicada')
    return {s: matrices[s].loc[:, retained].copy() for s in SEGMENTS}


def validate_final(bundle, woe, final, corr, retained, threshold):
    validate_correlation(corr, corr.columns.tolist())
    values=corr.loc[retained, retained].abs().to_numpy()
    if (values[np.triu_indices(len(retained), 1)] > threshold).any():
        raise ValueError('Pareja final supera el umbral')
    rows=[]
    for s in SEGMENTS:
        pd.testing.assert_frame_equal(final[s], woe[s].loc[:, retained], check_exact=True)
        if not final[s].index.equals(bundle['y'][s].index) or not final[s].index.equals(bundle['X'][s].index):
            raise ValueError(f'{s}: índices finales alterados')
        pd.testing.assert_series_equal(bundle['y'][s], bundle['metadata'][s].indicador_default, check_exact=True)
        if not np.isfinite(final[s].to_numpy(float)).all():raise ValueError('WoE final no finito')
        rows.append(dict(Segmento=s, Registros=len(final[s]), Variables=len(retained), Estado='VALIDADO'))
    return pd.DataFrame(rows)


def heatmap(corr, title):
    fig, ax = plt.subplots(figsize=(max(12, len(corr)*0.65), max(10, len(corr)*0.58)))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), annot=True, fmt='.2f',
        cmap='RdBu_r', center=0, vmin=-1, vmax=1, linewidths=.5, annot_kws={'size':7}, ax=ax)
    ax.set_title(title, fontweight='bold');fig.tight_layout()
    return fig


def plot_iv(decisions):
    fig, ax = plt.subplots(figsize=(13, max(7, len(decisions)*.35)))
    colors=decisions.Estado_correlacion.map({'CONSERVADA':'#287d8e','ELIMINADA_CORRELACION':'#c47755'})
    ax.barh(decisions.Variable, decisions.IV_final, color=colors)
    ax.invert_yaxis();ax.set_xlabel('IV final TRAIN');ax.set_title('Prioridad IV y selección por correlación')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#287d8e',label='Conservada'),Patch(color='#c47755',label='Eliminada por correlación')])
    for i,v in enumerate(decisions.IV_final):ax.text(v,i,f' {v:.4f}',va='center',fontsize=8)
    ax.margins(x=.12);fig.tight_layout();return fig


def save_selection(path, iv, decisions, retained, threshold, binning_path):
    records=json.loads(decisions.to_json(orient='records',force_ascii=False,double_precision=15))
    selection=dict(methodology_version='4.0', correlation_method='pearson', learned_on='TRAIN',
        threshold=float(threshold), comparison='abs(r) > threshold', iv_tie_break='Variable ascendente',
        reference_tie_break='Orden IV de las conservadas', iv_order=iv.Variable.tolist(),
        iv_final=dict(zip(iv.Variable,map(float,iv.IV_final))), retained_variables=list(retained),
        eliminated_variables=decisions.loc[decisions.Estado_correlacion.eq('ELIMINADA_CORRELACION'),'Variable'].tolist(),
        decisions=records, woe_artifact=str(Path(binning_path)), woe_artifact_sha256=sha256(binning_path))
    Path(path).write_text(json.dumps(selection,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return selection


def load_selection(path, binning_path=None):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    source=Path(binning_path or data['woe_artifact'])
    retained=data['retained_variables']; removed=data['eliminated_variables'];order=data['iv_order']
    if (data['methodology_version']!='4.0' or data['learned_on']!='TRAIN' or data['correlation_method']!='pearson'
            or sha256(source)!=data['woe_artifact_sha256'] or len(order)!=len(set(order))
            or set(retained)&set(removed) or set(retained+removed)!=set(order)
            or retained!=[v for v in order if v in retained] or not 0<=data['threshold']<=1):
        raise ValueError('Selección o hash WoE incompatible')
    return data


def reconstruct(bundle, selection, biv, binning_path=None):
    source=Path(binning_path or selection['woe_artifact'])
    if sha256(source)!=selection['woe_artifact_sha256']:raise ValueError('Hash WoE alterado')
    artifact=biv.load_binning(source)
    matrices,_=frozen_transform(bundle, artifact, biv)
    return apply_selection(matrices, selection['retained_variables'])


def export_results(root, bundle, final, iv, corr, decisions, details, retained, threshold, binning_path, figures):
    root=Path(root)
    for name in ('matrices','reportes','graficos','evidencias'):(root/name).mkdir(parents=True,exist_ok=True)
    frames={'df_iv_final':iv,'corr_woe_completa':corr,'corr_woe_absoluta':corr.abs(),
        'corr_woe_final':corr.loc[retained,retained], 'df_correlacion_decisiones':decisions,
        'df_correlaciones_descartadas':details,'variables_finales':iv.loc[iv.Variable.isin(retained)].reset_index(drop=True)}
    for name,frame in frames.items():frame.to_csv(root/f'reportes/{name}.csv',index=name.startswith('corr_'))
    indices={}
    for s,X in final.items():
        file=root/f'matrices/X_{s.lower()}_uncorr.csv';X.to_csv(file,index_label='_row_id')
        restored=pd.read_csv(file,index_col='_row_id',float_precision='round_trip');restored.index.name=X.index.name
        pd.testing.assert_frame_equal(X,restored,check_exact=True)
        indices[s]=dict(n_rows=len(X),index_name=X.index.name,index_dtype=str(X.index.dtype),
            row_ids=X.index.tolist(),target=bundle['y'][s].astype(int).tolist(),file=str(file),sha256=sha256(file),
            column_order=X.columns.tolist(),dtypes={c:str(t) for c,t in X.dtypes.items()})
    (root/'matrices/manifest_indices.json').write_text(json.dumps(indices,indent=2,allow_nan=False),encoding='utf-8')
    selection=save_selection(root/'seleccion_correlacion.json',iv,decisions,retained,threshold,binning_path)
    for name,fig in figures.items():fig.savefig(root/f'graficos/{name}.png',dpi=140,bbox_inches='tight')
    return selection
