"""Bivariado TRAIN-only. Importar mediante importlib.util por su prefijo 3.0.0.
CLI export-v4: persiste la seleccion PSI existente sin ejecutar filtros ni PSI.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy
import hashlib
import json
import math
import textwrap
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import spearmanr

ROOT = Path('4.ANALIS_UV_BV_MV/BIVARIADO')
PSI_ROOT = Path('4.ANALIS_UV_BV_MV/PSI_TEMPORAL')
SOURCE = Path('2.OUTPUT/1.1.0.TAB_SEGM_TRAIN_TEST_OOT.csv')
MISSING, NEW, POS_INF, NEG_INF = '__MISSING__', '__NEW__', '__POS_INF__', '__NEG_INF__'
SPECIAL = (MISSING, NEW, POS_INF, NEG_INF)
DEFAULTS = {
    'POLICY_VERSION': 'B.3', 'MIN_IV_TOTAL': .02, 'MIN_FIT_OBS': 400,
    'MIN_EVAL_BIN_N': 50,
    'Q_VALUES': list(range(2, 11)), 'ALPHA': .5,
    'LOW_CARDINALITY_MAX': 10, 'MAX_CATEGORIES': 50,
    'MIN_BIN_PCT': .05, 'MIN_BIN_N': 200, 'MIN_GOODS': 20, 'MIN_BADS': 10,
    'MIN_MONTH_BIN_N': 50, 'MIN_MONTH_BIN_EVENTS': 3,
    'MIN_TEMPORAL_COVERAGE': .70, 'TEMPORAL_Z_ALERT': 3.,
    'MAX_TEMPORAL_ALERT_FRACTION': .20, 'IV_RETAIN_FRACTION': .90,
    'APPROX_TREND_RHO': .80, 'MERGE_ADJACENT': True,
    'UNSEEN_POLICY': 'training_prior_woe_0',
    'MIN_PREFERRED_BINS': 3, 'MIN_IV_ORDINARY': .01,
    'MIN_RATE_GAP': .005, 'MIN_TWO_BIN_IV': .02,
    'MIN_TWO_BIN_AUC': .55, 'MIN_TWO_BIN_RATE_GAP': .01,
    'TWO_BIN_MAX_IV_LOSS': .10, 'MAX_COMPARISON_IV_LOSS': .20,
}


def now():
    return datetime.now(ZoneInfo('America/Lima')).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameters(**overrides):
    if set(overrides) - set(DEFAULTS):
        raise ValueError('Parametros desconocidos')
    result = {**copy.deepcopy(DEFAULTS), **overrides}
    if result['POLICY_VERSION'] not in ['B.2','B.3']:
        raise ValueError('POLICY_VERSION debe ser B.2 o B.3')
    if not 0 <= result['MIN_IV_TOTAL'] <= 1 or result['MIN_FIT_OBS'] < 2 or result['MIN_EVAL_BIN_N'] < 1:
        raise ValueError('Umbrales de IV/volumen invalidos')
    if not result['Q_VALUES'] or any(isinstance(q, bool) or not isinstance(q, int) or q < 2 for q in result['Q_VALUES']):
        raise ValueError('Q_VALUES requiere enteros >=2')
    for key in ['ALPHA', 'TEMPORAL_Z_ALERT']:
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f'{key} debe ser positivo y finito')
    for key in ['MIN_BIN_PCT', 'MIN_TEMPORAL_COVERAGE', 'IV_RETAIN_FRACTION', 'MAX_TEMPORAL_ALERT_FRACTION', 'APPROX_TREND_RHO']:
        if not 0 < result[key] <= 1:
            raise ValueError(f'{key} debe pertenecer a (0,1]')
    for key in ['LOW_CARDINALITY_MAX', 'MAX_CATEGORIES', 'MIN_BIN_N', 'MIN_GOODS', 'MIN_BADS', 'MIN_MONTH_BIN_N', 'MIN_MONTH_BIN_EVENTS']:
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f'{key} debe ser entero positivo')
    if result['UNSEEN_POLICY'] not in ['training_prior_woe_0', 'raise']:
        raise ValueError('Politica de categorias nuevas desconocida')
    for key in ['MIN_IV_ORDINARY', 'MIN_RATE_GAP', 'MIN_TWO_BIN_IV',
                'MIN_TWO_BIN_RATE_GAP', 'TWO_BIN_MAX_IV_LOSS', 'MAX_COMPARISON_IV_LOSS']:
        if not np.isfinite(result[key]) or not 0 <= result[key] <= 1:
            raise ValueError(f'{key} debe pertenecer a [0,1]')
    if result['POLICY_VERSION']=='B.2' and (not .5 <= result['MIN_TWO_BIN_AUC'] <= 1 or result['MIN_PREFERRED_BINS'] < 3):
        raise ValueError('Preferencia minima 3 bins; AUC de dos bins >=0.5')
    return result


def month_periods(values):
    s = pd.Series(values)
    text = s.astype('string').str.replace(r'\.0$', '', regex=True)
    compact = text.str.fullmatch(r'\d{6}').fillna(False)
    dates = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
    dates.loc[compact] = pd.to_datetime(text[compact], format='%Y%m', errors='raise')
    dates.loc[~compact] = pd.to_datetime(text[~compact], errors='raise')
    if dates.isna().any():
        raise ValueError('Periodo nulo')
    return dates.dt.to_period('M')


def _validate_target(index, target):
    if not index.is_unique or not index.equals(target.index):
        raise ValueError('Indices de predictores y target desalineados o duplicados')
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError('indicador_default debe ser binario 0/1, sin nulos')
    if target.nunique() != 2:
        raise ValueError('TRAIN necesita buenos y malos')


def export_v4_from_psi(source=SOURCE, psi_root=PSI_ROOT, root=ROOT):
    """Reconstruye v4 de decisiones persistidas; no recalcula ninguna regla."""
    source, psi_root, root = Path(source), Path(psi_root), Path(root)
    evidence_file = psi_root / 'evidencias/ejecucion.json'
    summary_file = psi_root / 'reportes/df_resumen_psi.csv'
    report_file = psi_root / 'reportes/df_reporte_univariado.csv'
    evidence = json.loads(evidence_file.read_text(encoding='utf-8'))
    if sha256(source) != evidence['source_sha256']:
        raise ValueError('La fuente no coincide con la ejecucion PSI persistida')
    if not evidence.get('v4_preserved') or evidence.get('configuration', {}).get('AUTO_DROP_PSI'):
        raise ValueError('Se requiere listado v4 persistido explicito si hubo exclusiones PSI')
    data = pd.read_csv(source)
    summary, report = pd.read_csv(summary_file), pd.read_csv(report_file)
    selected = summary.Variable.tolist()
    if summary.Variable.duplicated().any() or not set(selected).issubset(data.columns):
        raise ValueError('Seleccion PSI ambigua o inexistente en la fuente')
    if set(report.loc[report.Filtro.eq('PSI mensual'), 'Variable']) != set(selected):
        raise ValueError('Reporte y seleccion PSI no coinciden')
    if set(report.loc[report.Decision.eq('Excluir'), 'Variable']) & set(selected):
        raise ValueError('Variables seleccionadas figuran como excluidas')
    features = [c for c in data.columns if c in set(selected)]
    if len(features) != evidence.get('v4_columns', evidence['v3_columns']):
        raise ValueError('Numero de columnas distinto al contrato v4 persistido')
    auxiliary = ['cliente_id', 'periodo_observacion', 'SEGMENTO', 'indicador_default']
    if set(features) & set(auxiliary):
        raise ValueError('Auxiliares incluidos en predictores')
    if set(data.SEGMENTO.unique()) != {'TRAIN', 'TEST', 'OOT'}:
        raise ValueError('Segmentos inesperados')
    signatures = {'source': sha256(source), 'psi_summary': sha256(summary_file),
                  'psi_report': sha256(report_file), 'psi_execution': sha256(evidence_file)}
    version = hashlib.sha256(json.dumps(signatures, sort_keys=True).encode()).hexdigest()[:16]
    folder = root / 'entradas' / ('v4_' + version)
    manifest_file = folder / 'manifest.json'
    if manifest_file.exists():
        load_v4(manifest_file=manifest_file)
    else:
        folder.mkdir(parents=True, exist_ok=True)
        manifest = {'schema_version': 1, 'version': version, 'created_at': now(),
            'provenance': signatures, 'source_path': str(source),
            'selection_origin': 'PSI persistido: v4=v3, sin exclusiones fisicas',
            'filters_recalculated': False, 'psi_recalculated': False,
            'features': features, 'feature_dtypes': {c: str(data[c].dtype) for c in features},
            'target': 'indicador_default', 'segments': {}, 'files': {}}
        for segment in ['TRAIN', 'TEST', 'OOT']:
            sample = data.loc[data.SEGMENTO.eq(segment)]
            contents = {f'X_{segment.lower()}_v4.csv': sample[features],
                f'y_{segment.lower()}.csv': sample[['indicador_default']],
                f'metadata_{segment.lower()}.csv': sample[auxiliary]}
            for name, table in contents.items():
                table.to_csv(folder / name, index=True, index_label='_row_id')
                manifest['files'][name] = sha256(folder / name)
            manifest['segments'][segment] = {'n_rows': len(sample),
                'n_clients': int(sample.cliente_id.nunique()), 'n_defaults': int(sample.indicador_default.sum()),
                'first_period': str(month_periods(sample.periodo_observacion).min()),
                'last_period': str(month_periods(sample.periodo_observacion).max()),
                'metadata_dtypes': {c: str(sample[c].dtype) for c in auxiliary}}
        for name, original in [('df_resumen_psi.csv', summary_file), ('df_reporte_univariado.csv', report_file)]:
            (folder / name).write_bytes(original.read_bytes())
            manifest['files'][name] = sha256(folder / name)
        manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (root / 'entradas/LATEST.json').write_text(json.dumps({
        'manifest': str(manifest_file.relative_to(root / 'entradas')),
        'manifest_sha256': sha256(manifest_file)}, indent=2), encoding='utf-8')
    loaded = load_v4(manifest_file=manifest_file)
    for segment in ['TRAIN', 'TEST', 'OOT']:
        original = data.loc[data.SEGMENTO.eq(segment)]
        pd.testing.assert_frame_equal(loaded['X'][segment], original[features])
        pd.testing.assert_series_equal(loaded['y'][segment], original.indicador_default)
    return manifest_file


def load_v4(root=ROOT, manifest_file=None):
    """Carga artefactos sin leer la fuente original ni ejecutar notebooks."""
    root = Path(root)
    if manifest_file is None:
        pointer = json.loads((root / 'entradas/LATEST.json').read_text(encoding='utf-8'))
        manifest_file = root / 'entradas' / pointer['manifest']
        if sha256(manifest_file) != pointer['manifest_sha256']:
            raise ValueError('Manifest alterado')
    manifest_file = Path(manifest_file)
    folder = manifest_file.parent
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    if manifest['schema_version'] != 1:
        raise ValueError('Version de esquema no soportada')
    for name, digest in manifest['files'].items():
        file = folder / name
        if file.resolve().parent != folder.resolve() or sha256(file) != digest:
            raise ValueError(f'Archivo alterado o fuera de carpeta: {name}')
    bundle = {'X': {}, 'y': {}, 'metadata': {}, 'manifest': manifest,
              'manifest_file': str(manifest_file), 'psi_summary': pd.read_csv(folder / 'df_resumen_psi.csv')}
    for segment in ['TRAIN', 'TEST', 'OOT']:
        key = segment.lower()
        X = pd.read_csv(folder / f'X_{key}_v4.csv', index_col='_row_id', dtype=manifest['feature_dtypes'], float_precision='round_trip')
        y = pd.read_csv(folder / f'y_{key}.csv', index_col='_row_id')['indicador_default']
        metadata = pd.read_csv(folder / f'metadata_{key}.csv', index_col='_row_id', dtype=manifest['segments'][segment]['metadata_dtypes'])
        X.index.name = y.index.name = metadata.index.name = None
        if not X.index.is_unique or not X.index.equals(y.index) or not X.index.equals(metadata.index):
            raise ValueError('Indices persistidos desalineados')
        if list(X.columns) != manifest['features'] or len(X) != manifest['segments'][segment]['n_rows']:
            raise ValueError('Contrato de columnas/filas invalido')
        if y.isna().any() or not y.isin([0, 1]).all():
            raise ValueError('Target invalido')
        pd.testing.assert_series_equal(y, metadata.indicador_default)
        if not metadata.SEGMENTO.eq(segment).all() or metadata.cliente_id.isna().any():
            raise ValueError('Identificadores/segmentos invalidos')
        month_periods(metadata.periodo_observacion)
        bundle['X'][segment], bundle['y'][segment], bundle['metadata'][segment] = X, y, metadata
    combined = pd.Index(np.concatenate([bundle['X'][s].index for s in bundle['X']]))
    if not combined.is_unique:
        raise ValueError('Registros compartidos entre segmentos')
    if len(manifest['features']) != 29:
        warnings.warn(f"Se esperaban 29 variables; artefactos contienen {len(manifest['features'])}. No se fuerza el numero.")
    return bundle


def sample_diagnostics(bundle):
    records = []
    for segment, X in bundle['X'].items():
        metadata = bundle['metadata'][segment]
        months = month_periods(metadata.periodo_observacion)
        records.append({'Segmento': segment, 'Registros': len(X), 'Variables': X.shape[1],
            'Clientes': metadata.cliente_id.nunique(), 'Clientes_repetidos': metadata.cliente_id.duplicated().sum(),
            'Malos': int(bundle['y'][segment].sum()), 'Tasa_default': bundle['y'][segment].mean(),
            'Primer_periodo': str(months.min()), 'Ultimo_periodo': str(months.max()), 'Indices_target_segmento': 'VALIDADOS'})
    return pd.DataFrame(records)


def explore_train(X):
    records = []
    for name, s in X.items():
        counts = s.value_counts()
        numeric = pd.api.types.is_numeric_dtype(s)
        records.append({'Variable': name, 'Tipo': str(s.dtype), 'N_observaciones': len(s),
            'N_nulos': int(s.isna().sum()), 'Pct_nulos': 100 * s.isna().mean(), 'N_unicos': s.nunique(),
            'Pct_concentracion': 100 * counts.iloc[0] / len(s) if len(counts) else np.nan,
            'Pct_ceros': 100 * s.eq(0).mean() if numeric else np.nan,
            'N_infinitos': int(np.isinf(s).sum()) if numeric else 0})
    return pd.DataFrame(records)


def category_key(value, numeric=False):
    if numeric:
        return 'NUMBER:' + format(float(value), '.17g')
    return type(value).__name__ + ':' + json.dumps(value, ensure_ascii=False, default=str)


def _refresh_intervals(candidate):
    candidate['ordered_bins'] = [f'B{i:03d}' for i in range(len(candidate['edges']) - 1)]
    for label in list(candidate['bin_labels']):
        if label.startswith('B'):
            del candidate['bin_labels'][label]
    for i, label in enumerate(candidate['ordered_bins']):
        left, right = candidate['edges'][i:i + 2]
        candidate['bin_labels'][label] = f'({left:.8g}, {right:.8g}]'


def learn_candidate(s, q, config=None):
    """Aprende limites con X TRAIN; no recibe target ni TEST/OOT."""
    config = parameters() if config is None else config
    numeric = pd.api.types.is_numeric_dtype(s)
    valid = s[s.notna()]
    if numeric:
        valid = valid[np.isfinite(valid)]
    counts = valid.value_counts()
    result = {'kind': 'categorical', 'numeric': bool(numeric), 'q_requested': int(q),
        'method': 'categories', 'edges': [], 'category_groups': {}, 'bin_labels': {},
        'ordered_bins': [], 'fusion_history': [], 'reference_degenerate': len(counts) <= 1,
        'interval_convention': '(left, right], outer limits -infinity/+infinity',
        'missing_rule': 'dedicated __MISSING__; never imputed as zero',
        'infinity_rule': 'dedicated __NEG_INF__/__POS_INF__ quality bins'}
    if numeric and len(counts) > config['LOW_CARDINALITY_MAX']:
        result['kind'], result['method'] = 'intervals', 'quantiles'
        edges = np.unique(np.quantile(valid.to_numpy(float), np.linspace(0, 1, q + 1)))
        previous, retained = -np.inf, []
        for cut in edges[1:-1]:
            if ((valid > previous) & (valid <= cut)).any() and (valid > cut).any():
                retained.append(float(cut)); previous = cut
        result['edges'] = [-np.inf] + retained + [np.inf]
        _refresh_intervals(result)
    else:
        values = sorted(counts.index) if numeric else list(counts.index)
        if len(values) > config['MAX_CATEGORIES']:
            frequent = list(counts.index[:config['MAX_CATEGORIES']])
            for i, value in enumerate(frequent):
                label = f'C{i:03d}'
                result['category_groups'][category_key(value, numeric)] = label
                result['bin_labels'][label] = repr(value)
            for value in counts.index[config['MAX_CATEGORIES']:]:
                result['category_groups'][category_key(value, numeric)] = '__RARE__'
            result['bin_labels']['__RARE__'] = 'Categorias TRAIN poco frecuentes'
        else:
            for i, value in enumerate(values):
                label = f'C{i:03d}'
                result['category_groups'][category_key(value, numeric)] = label
                result['bin_labels'][label] = repr(value)
        if numeric:
            result['ordered_bins'] = list(result['bin_labels'])
    result['bin_labels'].update({MISSING: 'Ausencia de informacion', NEW: 'Categoria nueva',
        NEG_INF: '-infinito: anomalia', POS_INF: '+infinito: anomalia'})
    return result


def assign_bins(s, candidate):
    """Aplica limites exactos; no altera los valores ni los indices originales."""
    labels = pd.Series(MISSING, index=s.index, dtype=object)
    valid, values = s.notna(), s
    if candidate['numeric']:
        values = pd.to_numeric(s, errors='raise').astype(float)
        inf = valid & ~np.isfinite(values)
        labels.loc[inf & values.gt(0)] = POS_INF
        labels.loc[inf & values.lt(0)] = NEG_INF
        valid &= ~inf
    if candidate['kind'] == 'intervals':
        labels.loc[valid] = pd.cut(values.loc[valid], candidate['edges'], labels=False, right=True).map(lambda index: f'B{int(index):03d}')
    else:
        labels.loc[valid] = values.loc[valid].map(lambda value: candidate['category_groups'].get(category_key(value, candidate['numeric']), NEW))
    return labels


def generate_candidates(X_train, config=None):
    config = parameters() if config is None else config
    result = {}
    for variable, series in X_train.items():
        for q in config['Q_VALUES']:
            key = f'{variable}::q{q}'
            try:
                learned = learn_candidate(series, q, config)
            except Exception as error:
                if config.get('POLICY_VERSION')!='B.3': raise
                learned = _empty_configuration(q, str(error))
            result[key] = {'Variable': variable, 'Candidate_ID': key,
                'configuration': learned, 'Origen': 'CUANTILES'}
    return result


def candidate_inventory(candidates):
    return pd.DataFrame([{'Variable': item['Variable'], 'Candidate_ID': key,
        'q_solicitado': item['configuration']['q_requested'], 'Metodo': item['configuration']['method'],
        'Bins_definidos': len(item['configuration']['ordered_bins']) or len(set(item['configuration']['category_groups'].values())),
        'Cortes_exactos': repr(item['configuration']['edges']), 'Origen': item['Origen']}
        for key, item in candidates.items()])


def bin_statistics(labels, y, candidate, alpha=.5):
    _validate_target(labels.index, y)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('ALPHA debe ser positivo y finito')
    data = pd.DataFrame({'Bin': labels, 'y': y})
    stats = data.groupby('Bin', sort=False).y.agg(N='size', Malos='sum')
    stats['Buenos'] = stats.N - stats.Malos
    ordered = ordinary_order(candidate)
    order = ordered + [b for b in candidate['bin_labels'] if b not in ordered and b not in SPECIAL]
    order += [NEG_INF, POS_INF, NEW, MISSING]
    observed = [b for b in order if b in stats.index]
    observed += [b for b in stats.index if b not in observed]
    stats = stats.reindex(observed).reset_index()
    k = len(stats)
    stats['Dist_clientes'] = stats.N / len(labels)
    stats['Tasa_default'] = stats.Malos / stats.N
    stats['Dist_buenos_suavizada'] = (stats.Buenos + alpha) / (int((y == 0).sum()) + alpha * k)
    stats['Dist_malos_suavizada'] = (stats.Malos + alpha) / (int((y == 1).sum()) + alpha * k)
    stats['WoE'] = np.log(stats.Dist_buenos_suavizada / stats.Dist_malos_suavizada)
    stats['IV_bin'] = (stats.Dist_buenos_suavizada - stats.Dist_malos_suavizada) * stats.WoE
    stats['Es_especial'] = stats.Bin.isin(SPECIAL)
    stats['Es_missing'] = stats.Bin.eq(MISSING)
    stats['Exento_min_pct'] = stats.Es_missing
    stats['Bin_puro'] = stats.Buenos.eq(0) | stats.Malos.eq(0)
    stats['Etiqueta'] = stats.Bin.map(candidate['bin_labels'])
    stats['Limite_inferior'], stats['Limite_superior'] = np.nan, np.nan
    if candidate['kind'] == 'intervals':
        for i, label in enumerate(ordered):
            mask = stats.Bin.eq(label)
            stats.loc[mask, 'Limite_inferior'] = candidate['edges'][i]
            stats.loc[mask, 'Limite_superior'] = candidate['edges'][i + 1]
    elif candidate['numeric']:
        for label in ordered:
            values = [float(k[7:]) for k, v in candidate['category_groups'].items() if v == label]
            stats.loc[stats.Bin.eq(label), 'Limite_inferior'] = min(values)
            stats.loc[stats.Bin.eq(label), 'Limite_superior'] = max(values)
    return stats


def ordinary_order(candidate):
    """Numeric order from exact limits/memberships, never from visual labels."""
    if not candidate['numeric']:
        return []
    if candidate['kind'] == 'intervals':
        return [f'B{i:03d}' for i in range(len(candidate['edges']) - 1)]
    bounds = {}
    for key, group in candidate['category_groups'].items():
        if group not in SPECIAL and key.startswith('NUMBER:'):
            bounds.setdefault(group, []).append(float(key[7:]))
    return sorted(bounds, key=lambda group: min(bounds[group]))


def monotonicity(stats, candidate, config):
    # __MISSING__ and all other quality bins are explicitly excluded by code.
    order = [label for label in ordinary_order(candidate) if label not in SPECIAL]
    rates = stats.set_index('Bin').Tasa_default.reindex(order).dropna().to_numpy()
    if len(rates) < 2:
        return 'NO_EVALUABLE', np.nan
    delta = np.diff(rates)
    if np.all(np.abs(delta) <= 1e-12):
        return 'NO_EVALUABLE', 0.
    rho = float(spearmanr(np.arange(len(rates)), rates).statistic)
    if np.all(delta >= -1e-12): return 'CRECIENTE', rho
    if np.all(delta <= 1e-12): return 'DECRECIENTE', rho
    return ('TENDENCIA_APROXIMADA' if abs(rho) >= config['APPROX_TREND_RHO'] else 'NO_MONOTONA'), rho


def monthly_rates(labels, y, metadata, stats, config):
    if not metadata.index.equals(labels.index) or not metadata.SEGMENTO.eq('TRAIN').all():
        raise ValueError('Estabilidad solo con metadata TRAIN alineada')
    months = month_periods(metadata.periodo_observacion).astype(str)
    grouped = pd.DataFrame({'Periodo': months, 'Bin': labels, 'y': y}).groupby(['Periodo', 'Bin']).y.agg(N='size', Malos='sum')
    grid = pd.MultiIndex.from_product([sorted(months.unique()), stats.Bin], names=['Periodo', 'Bin'])
    result = grouped.reindex(grid, fill_value=0).reset_index()
    result['Buenos'] = result.N - result.Malos
    result['Tasa_default'] = result.Malos.div(result.N.where(result.N.gt(0)))
    result['Tasa_train_bin'] = result.Bin.map(stats.set_index('Bin').Tasa_default)
    pooled_n = result.Bin.map(stats.set_index('Bin').N)
    pooled_b = result.Bin.map(stats.set_index('Bin').Malos)
    other_n = pooled_n - result.N
    other_rate = (pooled_b - result.Malos).div(other_n.where(other_n.gt(0)))
    se = np.sqrt(other_rate * (1 - other_rate) * (1 / result.N.where(result.N.gt(0)) + 1 / other_n.where(other_n.gt(0))))
    result['Z_vs_otros_meses'] = (result.Tasa_default - other_rate).abs().div(se.where(se.gt(0)))
    n, rate, z = result.N.where(result.N.gt(0)), result.Tasa_default, 1.96
    center = (rate + z*z/(2*n)) / (1 + z*z/n)
    radius = z * np.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / (1 + z*z/n)
    result['IC95_inferior'], result['IC95_superior'] = center - radius, center + radius
    sufficient = result.N.ge(config['MIN_MONTH_BIN_N']) & result.Malos.ge(config['MIN_MONTH_BIN_EVENTS']) & result.Buenos.ge(config['MIN_MONTH_BIN_EVENTS'])
    result['Estado_mes_bin'] = np.where(result.N.eq(0), 'SIN_OBSERVACIONES', np.where(sufficient, 'EVALUABLE', 'VOLUMEN_INSUFICIENTE'))
    result['Alerta_tasa'] = sufficient & result.Z_vs_otros_meses.gt(config['TEMPORAL_Z_ALERT'])
    result['Diferencia_tasa_pp'] = 100 * (result.Tasa_default - result.Tasa_train_bin)
    return result


def _bad_bin_mask(stats, config):
    return (stats.N.lt(config['MIN_BIN_N']) | stats.Dist_clientes.lt(config['MIN_BIN_PCT']) |
            stats.Buenos.lt(config['MIN_GOODS']) | stats.Malos.lt(config['MIN_BADS']))


def _counts_metrics(n, bad, alpha):
    """Recompute every rate/WoE/IV from counts; never average old estimates."""
    n, bad = np.asarray(n, float), np.asarray(bad, float)
    good = n - bad
    pg = (good + alpha) / (good.sum() + alpha * len(n))
    pb = (bad + alpha) / (bad.sum() + alpha * len(n))
    woe = np.log(pg / pb)
    return good, bad / n, woe, (pg - pb) * woe


def _counts_auc(good, bad, score):
    good, bad, score = np.asarray(good), np.asarray(bad), np.asarray(score)
    if good.sum() == 0 or bad.sum() == 0:
        return np.nan
    _, groups = np.unique(score, return_inverse=True)
    g = np.bincount(groups, weights=good)
    b = np.bincount(groups, weights=bad)
    return float(np.sum(b * (np.cumsum(g) - g / 2)) / (g.sum() * b.sum()))


def _temporal_arrays(n, bad, monthly_n, monthly_bad, config):
    other_n = n[None, :] - monthly_n
    with np.errstate(divide='ignore', invalid='ignore'):
        other_rate = (bad[None, :] - monthly_bad) / other_n
        se = np.sqrt(other_rate * (1 - other_rate) * (1 / monthly_n + 1 / other_n))
        z = np.abs(monthly_bad / monthly_n - other_rate) / se
    enough = ((monthly_n >= config['MIN_MONTH_BIN_N']) &
              (monthly_bad >= config['MIN_MONTH_BIN_EVENTS']) &
              (monthly_n - monthly_bad >= config['MIN_MONTH_BIN_EVENTS']))
    coverage = float(enough.mean()) if enough.size else 0.
    fraction = float((z[enough] > config['TEMPORAL_Z_ALERT']).mean()) if enough.any() else np.nan
    return coverage, fraction


def _quality_issues(n, bad, iv_ordinary, rate_gap, auc_ordinary, monotone,
                    coverage, fraction, total, config, *, numeric=True,
                    intervals_valid=True, has_infinities=False, iv_total=None):
    issues = []
    if not intervals_valid: issues.append('CORTES_INVALIDOS')
    if len(n) < 2: issues.append('MENOS_DE_DOS_BINS_ORDINARIOS')
    if numeric and np.sum(n) < 2 * config['MIN_BIN_PCT'] * total - 1e-10:
        issues.append('NO_FACTIBLE_MASA_ORDINARIA_PARA_DOS_BINS')
    if np.any(np.asarray(n) / total < config['MIN_BIN_PCT'] - 1e-12):
        issues.append('ORDINARIOS_INFERIORES_MIN_PCT' if numeric else 'CATEGORIAS_NATURALES_ESCASAS')
    if (np.any(np.asarray(n) < config['MIN_BIN_N']) or
            np.any(np.asarray(bad) < config['MIN_BADS']) or
            np.any(np.asarray(n) - np.asarray(bad) < config['MIN_GOODS'])):
        issues.append('ORDINARIOS_CON_VOLUMEN_O_EVENTOS_INSUFICIENTES')
    if numeric and monotone not in ['CRECIENTE', 'DECRECIENTE']:
        issues.append('MONOTONICIDAD_ESTRICTA_NO_RESUELTA')
    b3 = config.get('POLICY_VERSION') == 'B.3'
    if b3 and (iv_total if iv_total is not None else iv_ordinary) < config['MIN_IV_TOTAL'] - 1e-12:
        issues.append('IV_TOTAL_INFERIOR_0_02')
    if not b3 and iv_ordinary < config['MIN_IV_ORDINARY'] - 1e-12:
        issues.append('IV_ORDINARIO_INSUFICIENTE')
    if rate_gap < config['MIN_RATE_GAP'] - 1e-12:
        issues.append('DIFERENCIACION_DE_RIESGO_INSUFICIENTE')
    if not b3 and len(n) == 2 and (iv_ordinary < config['MIN_TWO_BIN_IV'] - 1e-12 or
                        rate_gap < config['MIN_TWO_BIN_RATE_GAP'] - 1e-12 or
                        not np.isfinite(auc_ordinary) or auc_ordinary < config['MIN_TWO_BIN_AUC']):
        issues.append('DOS_BINS_SIN_JUSTIFICACION_ESTADISTICA')
    if coverage < config['MIN_TEMPORAL_COVERAGE']:
        issues.append('COBERTURA_TEMPORAL_INSUFICIENTE')
    if not np.isfinite(fraction) or fraction > config['MAX_TEMPORAL_ALERT_FRACTION']:
        issues.append('TASAS_MENSUALES_REQUIEREN_REVISION')
    if has_infinities: issues.append('INFINITOS_CALIDAD_DATOS')
    return issues


def _partition_counts(n, bad, chunks):
    return (np.array([n[a:b].sum() for a, b in chunks]),
            np.array([bad[a:b].sum() for a, b in chunks]))


def _partition_configuration(base, order, chunks):
    result = copy.deepcopy(base)
    result['fusion_history'] = []
    if base['kind'] == 'intervals':
        result['edges'] = [base['edges'][0]] + [base['edges'][b] for _, b in chunks[:-1]] + [base['edges'][-1]]
        _refresh_intervals(result)
    else:
        mapping = {label: f'G{i:03d}' for i, (a, b) in enumerate(chunks) for label in order[a:b]}
        result['category_groups'] = {key: mapping[label] for key, label in base['category_groups'].items()}
        result['ordered_bins'] = [f'G{i:03d}' for i in range(len(chunks))]
        result['bin_labels'] = {key: value for key, value in base['bin_labels'].items() if key in SPECIAL}
        for label in result['ordered_bins']:
            values = sorted(float(key[7:]) for key, group in result['category_groups'].items() if group == label)
            result['bin_labels'][label] = '{' + ', '.join(f'{v:g}' for v in values) + '}'
        result['method'] = 'numeric_adjacent_categories'
    result['partition_chunks'] = [list(chunk) for chunk in chunks]
    return result


def _fusion_history(base, order, chunks, n, bad, special_n, special_bad, total, config):
    """Trace only adjacent operations, recomputing counts and smoothed metrics."""
    current = [(i, i + 1) for i in range(len(order))]
    keep = {b for _, b in chunks[:-1]}
    history = []
    while len(current) > len(chunks):
        cn, cb = _partition_counts(n, bad, current)
        _, cr, _, iv_before = _counts_metrics(np.r_[cn, special_n], np.r_[cb, special_bad], config['ALPHA'])
        deficient = ((cn / total < config['MIN_BIN_PCT'] - 1e-12) | (cn < config['MIN_BIN_N']) |
                     (cb < config['MIN_BADS']) | (cn - cb < config['MIN_GOODS']))
        possible = [i for i in range(len(current) - 1) if current[i][1] not in keep]
        index = min(possible, key=lambda i: (not (deficient[i] or deficient[i+1]), abs(cr[i+1] - cr[i]), i))
        left, right = current[index:index+2]
        removed = left[1]
        next_chunks = current[:index] + [(left[0], right[1])] + current[index+2:]
        nn, nb = _partition_counts(n, bad, next_chunks)
        _, nr, nw, ni = _counts_metrics(np.r_[nn, special_n], np.r_[nb, special_bad], config['ALPHA'])
        history.append({'step': len(history)+1, 'source': 'TRAIN', 'adjacent': True,
            'left_original_bins': order[left[0]:left[1]], 'right_original_bins': order[right[0]:right[1]],
            'removed_cut': base['edges'][removed] if base['kind']=='intervals' else None,
            'reason': 'VOLUMEN_EVENTOS' if deficient[index] or deficient[index+1] else 'MONOTONICIDAD_SIMPLIFICACION',
            'N_after': int(nn[index]), 'Buenos_after': int(nn[index]-nb[index]), 'Malos_after': int(nb[index]),
            'Pct_after': float(nn[index]/total), 'Tasa_after': float(nr[index]), 'WoE_after': float(nw[index]),
            'IV_before': float(iv_before.sum()), 'IV_after': float(ni.sum()), 'missing_merged': False})
        current = next_chunks
    return history


def search_adjacent_partitions(s, y, candidate, config, metadata=None):
    """Enumerate every contiguous coarsening (<=2**9 for q<=10), both directions.

    Retain the best quality/IV partition for each direction and bin count. Each
    frontier member is evaluated again from raw TRAIN records downstream.
    """
    _validate_target(s.index, y)
    order = ordinary_order(candidate)
    audit = {'n_initial': len(order), 'partitions_examined': 0, 'feasible_partitions': 0,
             'directions': ['CRECIENTE', 'DECRECIENTE'], 'search': 'EXHAUSTIVE_CONTIGUOUS',
             'binary_or_nominal_natural': not candidate['numeric'] or len(order) <= 2}
    if not candidate['numeric'] or len(order) <= 2:
        return [], audit
    if len(order) > 12:
        raise ValueError('Busqueda exhaustiva limitada a 12 bins ordinarios; reduzca q/cardinalidad')
    labels = assign_bins(s, candidate)
    stats = bin_statistics(labels, y, candidate, config['ALPHA'])
    regular = stats.set_index('Bin').reindex(order)
    if regular.N.isna().any():
        return [], {**audit, 'reason': 'INTERVALO_SIN_OBSERVACIONES'}
    n, bad = regular.N.to_numpy(float), regular.Malos.to_numpy(float)
    special = stats[stats.Es_especial]
    sn, sb = special.N.to_numpy(float), special.Malos.to_numpy(float)
    monthly_n = monthly_bad = None
    if metadata is not None:
        if not metadata.index.equals(s.index) or not metadata.SEGMENTO.eq('TRAIN').all():
            raise ValueError('Fusiones exclusivamente con metadata TRAIN')
        temp = pd.DataFrame({'month': month_periods(metadata.periodo_observacion).astype(str), 'bin': labels, 'bad': y})
        counts = temp.groupby(['month', 'bin']).bad.agg(N='size', Bad='sum')
        monthly_n = counts.N.unstack('bin').reindex(columns=order).fillna(0).to_numpy(float)
        monthly_bad = counts.Bad.unstack('bin').reindex(columns=order).fillna(0).to_numpy(float)
    frontier = {}
    has_infinities = bool(labels.isin([POS_INF, NEG_INF]).any())
    for mask in range(1, 1 << (len(order)-1)):
        kept = [i+1 for i in range(len(order)-1) if mask & (1 << i)]
        borders = [0] + kept + [len(order)]
        chunks = list(zip(borders[:-1], borders[1:]))
        if len(chunks) == len(order):
            continue  # The original q candidate is already retained unconditionally.
        audit['partitions_examined'] += 1
        cn, cb = _partition_counts(n, bad, chunks)
        good, rates, woe, iv = _counts_metrics(np.r_[cn, sn], np.r_[cb, sb], config['ALPHA'])
        k = len(chunks); rates = rates[:k]
        delta = np.diff(rates)
        direction = ('CRECIENTE' if np.all(delta >= -1e-12) and np.ptp(rates)>1e-12
                     else 'DECRECIENTE' if np.all(delta <= 1e-12) and np.ptp(rates)>1e-12 else 'NO_MONOTONA')
        if monthly_n is None:
            coverage, fraction = 1., 0.  # Compatibility helper, not approval/selection API.
        else:
            mn = np.column_stack([monthly_n[:,a:b].sum(axis=1) for a,b in chunks])
            mb = np.column_stack([monthly_bad[:,a:b].sum(axis=1) for a,b in chunks])
            coverage, fraction = _temporal_arrays(cn, cb, mn, mb, config)
        auc = _counts_auc(good[:k], cb, -woe[:k])
        issues = _quality_issues(cn, cb, float(iv[:k].sum()), float(np.ptp(rates)), auc,
            direction, coverage, fraction, len(s), config,
            has_infinities=has_infinities, iv_total=float(iv.sum()))
        if not issues: audit['feasible_partitions'] += 1
        # No greedy local move: every contiguous partition competes in its bucket.
        bucket = (direction, k)
        score = (len(issues), -float(iv.sum()), fraction if np.isfinite(fraction) else 1., tuple(kept))
        if bucket not in frontier or score < frontier[bucket][0]:
            frontier[bucket] = (score, chunks)
    variants = []
    for (direction, _), (_, chunks) in sorted(frontier.items()):
        result = _partition_configuration(candidate, order, chunks)
        result['fusion_history'] = _fusion_history(candidate, order, chunks, n, bad, sn, sb, len(s), config)
        result['search_direction'] = direction
        result['search_audit'] = copy.deepcopy(audit)
        variants.append(result)
    return variants, audit


def merge_adjacent(s, y, candidate, config):
    """Compatibility helper: best contiguous coarsening, never touching nulos."""
    if config.get('POLICY_VERSION')=='B.3':
        stats=bin_statistics(assign_bins(s,candidate),y,candidate,config['ALPHA'])
        regular=stats[~stats.Es_especial]
        trend,_=monotonicity(stats,candidate,config)
        if not _quality_issues(regular.N.to_numpy(),regular.Malos.to_numpy(),regular.IV_bin.sum(),
                float(regular.Tasa_default.max()-regular.Tasa_default.min()) if len(regular) else 0.,
                np.nan,trend,1.,0.,len(s),config,numeric=candidate['numeric'],iv_total=stats.IV_bin.sum()):
            return copy.deepcopy(candidate)
    variants, _ = search_adjacent_partitions(s, y, candidate, config)
    if not variants:
        return copy.deepcopy(candidate)
    def rank(value):
        stats = bin_statistics(assign_bins(s, value), y, value, config['ALPHA'])
        ordinary = stats[~stats.Es_especial]
        trend, _ = monotonicity(stats, value, config)
        return (int(_bad_bin_mask(ordinary, config).sum()),
                trend not in ['CRECIENTE', 'DECRECIENTE'],
                (len(ordinary) < config['MIN_PREFERRED_BINS']) if config.get('POLICY_VERSION')=='B.2' else False,
                -stats.IV_bin.sum())
    return min(variants, key=rank)


DETAIL_COLUMNS=['Variable','Candidate_ID','Bin','N','Malos','Buenos','Dist_clientes',
    'Tasa_default','WoE','IV_bin','Es_especial','Es_missing','Exento_min_pct','Etiqueta',
    'Limite_inferior','Limite_superior','Volumen_insuficiente']
MONTH_COLUMNS=['Variable','Candidate_ID','Periodo','Bin','N','Malos','Buenos',
    'Tasa_default','Estado_mes_bin','Alerta_tasa','IC95_inferior','IC95_superior']


def _empty_configuration(q, error):
    return dict(kind='categorical',numeric=True,q_requested=q,method='ERROR_TECNICO',
        edges=[],category_groups={},bin_labels={MISSING:'Ausencia de informacion'},
        ordered_bins=[],fusion_history=[],technical_error=error)


def _error_row(key, item, error):
    return dict(Variable=item['Variable'],Candidate_ID=key,Origen=item['Origen'],
        q_solicitado=item['configuration']['q_requested'],Metodo='NO_EVALUABLE',
        Orden_numerico=item['configuration'].get('numeric',True),N_bins_efectivos=0,N_bins_total=0,
        N_bins_insuficientes=0,N_bins_puros=0,N_bins_ordinarios_insuficientes=0,
        Min_dist_clientes=np.nan,Min_pct_ordinario=np.nan,Cumple_5pct=False,
        IV_total=np.nan,IV_ordinario=np.nan,IV_missing=np.nan,AUC_ordinario=np.nan,
        Gap_tasas_ordinarias=np.nan,AUC_train_desarrollo=np.nan,Gini_train_desarrollo=np.nan,
        KS_train_desarrollo=np.nan,Monotonicidad='NO_EVALUABLE',Monotonicidad_estricta=False,
        Spearman_tasas_orden=np.nan,Cobertura_temporal=0.,Fraccion_alertas_tasas=np.nan,
        Calidad_ok=False,N_motivos_revision=999,Estado_candidato='NO_EVALUABLE',
        Motivos=error,Observaciones='No se imputa IV=0 ante un error',
        Cortes_exactos=repr(item['configuration'].get('edges',[])),N_fusiones=0,
        Alerta_missing_escaso=False,Cortes_validos=False,Min_buenos=0,Min_malos=0,
        N_particiones_exploradas=0,N_particiones_factibles=0)


def evaluate_candidates(X_train,y_train,metadata_train,candidates,config=None):
    """FIT only: target-independent generation plus TRAIN search/statistics."""
    config=parameters() if config is None else config
    if not X_train.index.equals(metadata_train.index) or not metadata_train.SEGMENTO.eq('TRAIN').all():
        raise ValueError('FIT exclusivamente con metadata TRAIN alineada')
    if not X_train.index.is_unique or not X_train.index.equals(y_train.index):
        raise ValueError('Indices de predictores y target desalineados o duplicados')
    if config.get('POLICY_VERSION')=='B.3':
        try:
            _validate_target(X_train.index,y_train)
            if len(X_train)<config['MIN_FIT_OBS']:
                raise ValueError('DATOS_INSUFICIENTES: TRAIN menor a MIN_FIT_OBS')
        except ValueError as error:
            return (pd.DataFrame([_error_row(k,v,str(error)) for k,v in candidates.items()]),
                pd.DataFrame(columns=DETAIL_COLUMNS),pd.DataFrame(columns=MONTH_COLUMNS),copy.deepcopy(candidates))
    return _evaluate_candidates_core(X_train,y_train,metadata_train,candidates,config)


def _evaluate_candidates_core(X_train, y_train, metadata_train, candidates, config=None):
    config = parameters() if config is None else config
    _validate_target(X_train.index, y_train)
    if not X_train.index.equals(metadata_train.index) or not metadata_train.SEGMENTO.eq('TRAIN').all():
        raise ValueError('El aprendizaje acepta exclusivamente TRAIN alineado')
    registry = copy.deepcopy(candidates)
    if config['MERGE_ADJACENT']:
        for key, item in list(registry.items()):
            if item['configuration'].get('technical_error'): continue
            try:
                variants, audit = search_adjacent_partitions(
                    X_train[item['Variable']], y_train, item['configuration'], config, metadata_train)
            except Exception as error:
                if config.get('POLICY_VERSION')!='B.3': raise
                item['configuration']['technical_error']=type(error).__name__+': '+str(error)
                continue
            item['configuration']['search_audit'] = audit
            for index, merged in enumerate(variants, 1):
                merged_key = f'{key}::B2_{index:02d}'
                registry[merged_key] = {**item, 'Candidate_ID': merged_key,
                    'configuration': merged, 'Origen': 'FUSION_ADYACENTE_TRAIN'}
    rows, details, stability = [], [], []
    for key, item in registry.items():
        try:
            if item['configuration'].get('technical_error'):
                raise ValueError(item['configuration']['technical_error'])
            variable, candidate = item['Variable'], item['configuration']
            labels = assign_bins(X_train[variable], candidate)
            stats = bin_statistics(labels, y_train, candidate, config['ALPHA'])
            rates = monthly_rates(labels, y_train, metadata_train, stats, config)
            stats['Volumen_insuficiente'] = _bad_bin_mask(stats, config)
            monotone, rho = monotonicity(stats, candidate, config)
            score = -labels.map(stats.set_index('Bin').WoE).astype(float)
            auc = float(roc_auc_score(y_train, score))
            fpr, tpr, _ = roc_curve(y_train, score)
            regular = stats[~stats.Es_especial]
            normal_rates = rates[rates.Bin.isin(regular.Bin)]
            evaluable = normal_rates.Estado_mes_bin.eq('EVALUABLE')
            temporal_coverage = float(evaluable.mean()) if len(normal_rates) else 0.
            alert_fraction = float(normal_rates.loc[evaluable, 'Alerta_tasa'].mean()) if evaluable.any() else np.nan
            n = regular.N.to_numpy(float)
            bad = regular.Malos.to_numpy(float)
            iv_ordinary = float(regular.IV_bin.sum())
            rate_gap = float(regular.Tasa_default.max() - regular.Tasa_default.min()) if len(regular) else 0.
            auc_ordinary = _counts_auc(regular.Buenos, regular.Malos, -regular.WoE)
            intervals_valid = (candidate['kind'] != 'intervals' or
                (candidate['edges'][0] == -np.inf and candidate['edges'][-1] == np.inf and
                 np.all(np.diff(candidate['edges']) > 0) and len(regular) == len(candidate['edges'])-1))
            issues = _quality_issues(n, bad, iv_ordinary, rate_gap, auc_ordinary,
                monotone, temporal_coverage, alert_fraction, len(X_train), config,
                numeric=candidate['numeric'], intervals_valid=intervals_valid,
                has_infinities=bool(labels.isin([POS_INF, NEG_INF]).any()), iv_total=float(stats.IV_bin.sum()))
            observations = []
            if monotone in ['NO_MONOTONA', 'TENDENCIA_APROXIMADA']: observations.append('TENDENCIA_NO_ESTRICTA_REQUIERE_REVISION_SIN_DESCARTE')
            missing_sparse = bool(stats.loc[stats.Es_missing, 'Volumen_insuficiente'].any())
            if missing_sparse: observations.append('MISSING_ESCASO_EXENTO_TAMANO_MANTENER_MONITOREO_WOE_SUAVIZADO')
            iv = float(stats.IV_bin.sum())
            if iv > .30: observations.append('IV_ALTO_REVISAR_FUGA_NO_DESCARTAR_POR_UMBRAL')
            rows.append({'Variable': variable, 'Candidate_ID': key, 'q_solicitado': candidate['q_requested'],
                'Origen': item['Origen'], 'Metodo': candidate['method'], 'Orden_numerico': candidate['numeric'], 'N_bins_efectivos': len(regular), 'N_bins_total': len(stats),
                'N_bins_insuficientes': int(stats.Volumen_insuficiente.sum()), 'N_bins_puros': int(stats.Bin_puro.sum()),
                'N_bins_ordinarios_insuficientes': int(_bad_bin_mask(regular, config).sum()),
                'Min_dist_clientes': float(stats.Dist_clientes.min()),
                'Min_pct_ordinario': float(regular.Dist_clientes.min()) if len(regular) else np.nan,
                'Cumple_5pct': bool(len(regular) and regular.Dist_clientes.ge(config['MIN_BIN_PCT']-1e-12).all()),
                'IV_ordinario': iv_ordinary, 'IV_missing': float(stats.loc[stats.Es_missing, 'IV_bin'].sum()),
                'Gap_tasas_ordinarias': rate_gap, 'AUC_ordinario': auc_ordinary,
                'Alerta_missing_escaso': missing_sparse, 'Cortes_validos': bool(intervals_valid),
                'Monotonicidad_estricta': monotone in ['CRECIENTE', 'DECRECIENTE'],
                'N_particiones_exploradas': candidate.get('search_audit', {}).get('partitions_examined', 0),
                'N_particiones_factibles': candidate.get('search_audit', {}).get('feasible_partitions', 0), 'Min_buenos': int(stats.Buenos.min()), 'Min_malos': int(stats.Malos.min()),
                'IV_total': iv, 'AUC_train_desarrollo': auc, 'Gini_train_desarrollo': 2*auc-1, 'KS_train_desarrollo': float(np.max(np.abs(tpr-fpr))),
                'Monotonicidad': monotone, 'Spearman_tasas_orden': rho, 'Cobertura_temporal': temporal_coverage,
                'Fraccion_alertas_tasas': alert_fraction, 'Calidad_ok': not issues, 'N_motivos_revision': len(issues),
                'Estado_candidato': ('NO_EVALUABLE' if len(regular)==0 else
                    'APROBADA' if not issues else 'RECHAZADA') if config.get('POLICY_VERSION')=='B.3'
                    else ('ACEPTABLE' if not issues else 'REQUIERE_REVISION'),
                'Motivos': '; '.join(issues) or 'Calidad y cobertura suficientes en TRAIN', 'Observaciones': '; '.join(observations),
                'Cortes_exactos': repr(candidate['edges']), 'N_fusiones': len(candidate['fusion_history'])})
            stats.insert(0, 'Candidate_ID', key); stats.insert(0, 'Variable', variable)
            rates.insert(0, 'Candidate_ID', key); rates.insert(0, 'Variable', variable)
            details.append(stats); stability.append(rates)
        except Exception as error:
            if config.get('POLICY_VERSION')!='B.3': raise
            rows.append(_error_row(key,item,type(error).__name__+': '+str(error)))
    return (pd.DataFrame(rows),
            pd.concat(details,ignore_index=True) if details else pd.DataFrame(columns=DETAIL_COLUMNS),
            pd.concat(stability,ignore_index=True) if stability else pd.DataFrame(columns=MONTH_COLUMNS),registry)


def select_configurations(candidates, details, registry, metadata_train, config=None):
    config = parameters() if config is None else config
    if not metadata_train.SEGMENTO.eq('TRAIN').all(): raise ValueError('Seleccion exclusivamente TRAIN')
    results, models = [], {}
    for variable, group in candidates.groupby('Variable', sort=False):
        eligible = group[group.Calidad_ok]
        pool = eligible if not eligible.empty else group[group.N_motivos_revision.eq(group.N_motivos_revision.min())]
        # Initially prefer >=3 ordinary bins. Two bins compete only with explicit
        # risk/IV/AUC justification (quality gate) and <=10% IV loss vs viable >=3.
        preferred = (pool[pool.N_bins_efectivos >= config['MIN_PREFERRED_BINS']]
                     if config.get('POLICY_VERSION') == 'B.2' else pool.iloc[:0])
        if not preferred.empty:
            two = pool[(pool.N_bins_efectivos == 2) &
                       (pool.IV_total >= (1-config['TWO_BIN_MAX_IV_LOSS']) * preferred.IV_total.max())]
            pool = pd.concat([preferred, two])
        iv_max = pool.IV_total.max()
        near = (pool.copy() if pool.IV_total.isna().all() else
                pool[pool.IV_total >= config['IV_RETAIN_FRACTION'] * iv_max].copy())
        near['Trend_penalty'] = near.Monotonicidad.map(lambda v: 0 if v in ['CRECIENTE', 'DECRECIENTE', 'NO_APLICA_NOMINAL'] else 1)
        chosen = near.sort_values(['N_bins_efectivos', 'Trend_penalty', 'Fraccion_alertas_tasas', 'IV_total', 'q_solicitado', 'Candidate_ID'],
            ascending=[True, True, True, False, True, True]).iloc[0]
        key = chosen.Candidate_ID
        candidate = copy.deepcopy(registry[key]['configuration'])
        stats = details[details.Candidate_ID.eq(key)].copy()
        state = ('APROBADA' if chosen.Calidad_ok else
                 ('NO_EVALUABLE' if group.Estado_candidato.eq('NO_EVALUABLE').all() else 'RECHAZADA')
                 if config.get('POLICY_VERSION') == 'B.3' else 'REQUIERE_REVISION')
        justification_two = ('Dos bins: volumen/eventos, IV ordinario, gap y AUC suficientes; '
            'sin alternativa >=3 viable o perdida IV dentro del margen configurado.') if chosen.N_bins_efectivos==2 else ''
        reason = (f"B.2: calidad y monotonicidad ordinaria TRAIN; preferencia >=3 bins; "
            f"IV >= {config['IV_RETAIN_FRACTION']:.0%} del maximo elegible; simplicidad y estabilidad. "
            + justification_two + ' ' + str(chosen.Motivos))
        if config.get('POLICY_VERSION') == 'B.3':
            justification_two = 'Dos bins validos: sin penalizacion ni filtro AUC adicional.' if chosen.N_bins_efectivos==2 else ''
            reason = ('B.3: >=2 bins ordinarios; IV total>=0.02; monotonicidad y volumen/eventos; '
                      'diferenciacion valida. Simplicidad dentro del margen de IV, sin preferencia >=3. '
                      + str(chosen.Motivos))
        row = chosen.drop(labels=['Trend_penalty']).to_dict()
        row.update(Estado_bivariado=state, Motivo_seleccion=reason,
            IV_max_candidatos=float(group.IV_total.max()), IV_max_grupo_calidad=float(iv_max),
            Justificacion_dos_bins=justification_two,
            Tiene_solucion_factible=bool(group.Calidad_ok.any()))
        results.append(row)
        candidate.update({'candidate_id': key, 'woe_by_bin': dict(zip(stats.Bin, stats.WoE.astype(float))),
            'iv_total': float(chosen.IV_total), 'state': state, 'selection_reason': reason,
            'bin_statistics': stats.to_dict('records'), 'unseen_policy': config['UNSEEN_POLICY'], 'unseen_woe': 0.,
            'unseen_explanation': 'Prior TRAIN neutral, no estimacion de grupo observado; cada uso se reporta.',
            'trained_on': 'TRAIN', 'reference_periods': sorted(month_periods(metadata_train.periodo_observacion).astype(str).unique().tolist())})
        candidate.update(methodology_version=config['POLICY_VERSION'], effective_bins=int(chosen.N_bins_efectivos),
            ordinary_order=ordinary_order(candidate), missing_bin=MISSING,
            missing_excluded_from_monotonicity=True, missing_exempt_min_pct=True,
            missing_statistics=stats.loc[stats.Bin.eq(MISSING)].to_dict('records'),
            new_cuts={'edges':copy.deepcopy(candidate['edges']),
                      'category_groups':copy.deepcopy(candidate['category_groups'])})
        models[variable] = candidate
    artifact = {'schema_version': 3 if config['POLICY_VERSION']=='B.3' else 2, 'methodology_version':config['POLICY_VERSION'], 'created_at': now(), 'parameters': copy.deepcopy(config),
        'target': 'indicador_default', 'target_definition': {'0': 'bueno', '1': 'malo'},
        'woe_definition': 'ln(P(bin|good)/P(bin|bad)); additive pseudocount on observed TRAIN bins',
        'iv_definition': 'sum((P_good-P_bad)*WoE); unscaled',
        'diagnostic_score': '-WoE; higher risk; apparent TRAIN development metrics', 'variables': models}
    if config['POLICY_VERSION']=='B.3':
        artifact['approved_variables']=[name for name,cfg in models.items() if cfg['state']=='APROBADA']
        artifact['rejected_variables']=[name for name,cfg in models.items() if cfg['state']=='RECHAZADA']
        artifact['not_evaluable_variables']=[name for name,cfg in models.items() if cfg['state']=='NO_EVALUABLE']
    return pd.DataFrame(results).sort_values('IV_total', ascending=False).reset_index(drop=True), artifact


def compare_binning(summary, details, artifact, baseline=None):
    """Historical comparison only; the baseline never influences fitting."""
    if baseline is None:
        pointer_file = 'B3_base.json' if artifact.get('methodology_version')=='B.3' else 'B2_base.json'
        pointer = json.loads((ROOT/'evidencias'/pointer_file).read_text(encoding='utf-8'))
        baseline = Path(pointer['backup'])
    baseline = Path(baseline)
    old_summary = pd.read_csv(baseline/'reportes/df_resumen_bivariado.csv').set_index('Variable')
    old_details = pd.read_csv(baseline/'reportes/df_cortes_seleccionados.csv')
    old_artifact = load_binning(baseline/'binning_woe_train.json')
    artifact['parent_artifact_sha256'] = sha256(baseline/'binning_woe_train.json')
    artifact['parent_artifact_path'] = str(baseline/'binning_woe_train.json')
    rows = []
    for current in summary.to_dict('records'):
        name = current['Variable']
        old = old_summary.loc[name] if name in old_summary.index else None
        before = old_details[old_details.Variable.eq(name) & ~old_details.Bin.isin(SPECIAL)]
        after = details[details.Candidate_ID.eq(current['Candidate_ID']) & ~details.Bin.isin(SPECIAL)]
        previous_model = old_artifact['variables'].get(name, {})
        artifact['variables'][name]['previous_cuts'] = {
            'edges': copy.deepcopy(previous_model.get('edges', [])),
            'category_groups': copy.deepcopy(previous_model.get('category_groups', {})),
            'candidate_id': previous_model.get('candidate_id'),
            'iv_total': previous_model.get('iv_total'), 'state': previous_model.get('state')}
        old_iv = float(old.IV_total) if old is not None else np.nan
        loss = (old_iv-current['IV_total'])/old_iv if old_iv>0 else np.nan
        if np.isfinite(loss) and abs(loss)<1e-12: loss=0.
        reasons = [current['Motivos'], current['Motivo_seleccion']]
        if current['Alerta_missing_escaso']:
            reasons.append('Missing aislado/exento de volumen; conserva IV y WoE con alerta de estimacion limitada')
        if not current['Tiene_solucion_factible']:
            reasons.append('Ninguna configuracion examinada cumple todos los controles '+artifact.get('methodology_version','B.2'))
        rows.append({'Variable':name,
            'Estado_anterior':old.Estado_bivariado if old is not None else 'SIN_ANTECEDENTE',
            'Estado_nuevo':current['Estado_bivariado'], 'IV_anterior':old_iv, 'IV_nuevo':current['IV_total'],
            'Bins_anteriores':int(old.N_bins_efectivos) if old is not None else np.nan,
            'Bins_finales':current['N_bins_efectivos'], 'Numero_fusiones':current['N_fusiones'],
            'Monotonicidad_anterior':old.Monotonicidad if old is not None else 'SIN_ANTECEDENTE',
            'Monotonicidad_nueva':current['Monotonicidad'],
            'Bin_minimo_anterior':before.Dist_clientes.min(), 'Bin_minimo_final':after.Dist_clientes.min(),
            'Cumple_5pct':current['Cumple_5pct'], 'Perdida_IV_pct':100*loss,
            'Perdida_IV_relevante':bool(np.isfinite(loss) and loss>artifact['parameters']['MAX_COMPARISON_IV_LOSS']),
            'Tiene_solucion_factible':current['Tiene_solucion_factible'],
            'Anteriormente_en_revision':bool(old is not None and old.Estado_bivariado=='REQUIERE_REVISION'),
            'Origen_fusiones':f"Candidato q={current['q_solicitado']}; fusiones desde ese candidato, no necesariamente desde cortes B.1",
            'Motivo_del_cambio':'; '.join(reasons)})
    return pd.DataFrame(rows)


def transform_woe(X, artifact, approved_only=None):
    """Aplica mapas congelados TRAIN; no recibe target ni reajusta nada."""
    models = artifact['variables']
    if approved_only is None:
        approved_only=artifact.get('methodology_version')=='B.3'
    columns=artifact.get('approved_variables',[]) if approved_only else list(X.columns)
    if not set(columns).issubset(X.columns) or not set(X.columns).issubset(models):
        raise ValueError('Columnas incompatibles con artefacto WoE')
    if not approved_only and set(X.columns)!=set(models):
        raise ValueError('Auditoria completa exige todas las variables del artefacto')
    output, diagnostics = pd.DataFrame(index=X.index), []
    for variable in columns:
        candidate = models[variable]
        if candidate.get('state')=='NO_EVALUABLE':
            raise ValueError(f'No existe transformacion estimada para {variable}')
        labels = assign_bins(X[variable], candidate)
        trained = labels.isin(candidate['woe_by_bin'])
        if not trained.all() and candidate['unseen_policy'] == 'raise': raise ValueError(f'Bin sin estimacion TRAIN: {variable}')
        output[variable] = labels.map(candidate['woe_by_bin']).fillna(candidate['unseen_woe']).astype(float)
        diagnostics.append({'Variable': variable, 'N_registros': len(X), 'N_missing': int(labels.eq(MISSING).sum()),
            'N_categorias_nuevas': int(labels.eq(NEW).sum()), 'N_infinitos': int(labels.isin([NEG_INF, POS_INF]).sum()),
            'N_bins_sin_estimacion_train': int((~trained).sum()), 'Politica_no_vistos': candidate['unseen_policy'],
            'Valores_WoE_finitos': bool(np.isfinite(output[variable]).all())})
    return output, pd.DataFrame(diagnostics,columns=['Variable','N_registros','N_missing',
        'N_categorias_nuevas','N_infinitos','N_bins_sin_estimacion_train','Politica_no_vistos','Valores_WoE_finitos'])


def validate_transformed(originals, transformed, targets, metadata, approved_columns=None):
    rows = []
    for segment, X in originals.items():
        result = transformed[segment]
        expected=X.columns if approved_columns is None else pd.Index(approved_columns)
        if not X.index.equals(result.index) or not expected.equals(result.columns): raise ValueError('Contrato WoE roto')
        if not X.index.equals(targets[segment].index) or not X.index.equals(metadata[segment].index): raise ValueError('Target/metadata desalineados')
        pd.testing.assert_series_equal(targets[segment], metadata[segment].indicador_default)
        if not np.isfinite(result.to_numpy(float)).all(): raise ValueError('WoE no finito')
        if not metadata[segment].SEGMENTO.eq(segment).all(): raise ValueError('Segmento alterado')
        rows.append({'Segmento': segment, 'Filas_originales': len(X), 'Filas_WoE': len(result),
                     'Variables_originales':X.shape[1],'Variables_WoE_aprobadas':result.shape[1], 'Estado': 'VALIDADO'})
    return pd.DataFrame(rows)


def _json_safe(value):
    if isinstance(value, dict): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)): return value.item()
    if isinstance(value, (float, np.floating)):
        if np.isnan(value): return None
        if np.isposinf(value): return '+Infinity'
        if np.isneginf(value): return '-Infinity'
        return float(value)
    return value


def save_binning(artifact, path):
    path = Path(path)
    path.write_text(json.dumps(_json_safe(artifact), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return path


def load_binning(path):
    artifact = json.loads(Path(path).read_text(encoding='utf-8'))
    if artifact.get('schema_version') not in [1,2,3]: raise ValueError('Esquema WoE no soportado')
    for candidate in artifact['variables'].values():
        candidate['edges'] = [(-np.inf if x == '-Infinity' else np.inf if x == '+Infinity' else x) for x in candidate['edges']]
        if candidate.get('state')!='NO_EVALUABLE' and candidate['kind'] == 'intervals' and not np.all(np.diff(candidate['edges']) > 0): raise ValueError('Limites no crecientes')
        if not all(np.isfinite(v) for v in candidate['woe_by_bin'].values()): raise ValueError('WoE no finito')
    if artifact.get('methodology_version')=='B.3':
        expected=[v for v,cfg in artifact['variables'].items() if cfg['state']=='APROBADA']
        if artifact.get('approved_variables')!=expected:
            raise ValueError('Lista aprobada inconsistente con estados congelados TRAIN')
    return artifact


def evaluate_frozen_samples(bundle,artifact):
    """Evaluation only: observed counts/rates, never quantiles/fusions/new WoE."""
    variables=artifact.get('approved_variables',
        [v for v,cfg in artifact['variables'].items() if cfg.get('state')=='APROBADA'])
    config=artifact['parameters']
    segments=('TRAIN','TEST','OOT')
    for segment in segments:
        X,y=bundle['X'][segment],bundle['y'][segment]
        if not X.index.equals(y.index) or not X.index.is_unique:
            raise ValueError('Evaluacion con indices/target desalineados')
        if y.isna().any() or not y.isin([0,1]).all():
            raise ValueError('Target observado invalido')
    rows,aggregates=[],[]
    for variable in variables:
        cfg=artifact['variables'][variable]
        labels={s:assign_bins(bundle['X'][s][variable],cfg) for s in segments}
        order=ordinary_order(cfg)
        order += [b for b in cfg['bin_labels'] if b not in order and b not in SPECIAL]
        # Predefined quality buckets are only activated for reporting, not fitted.
        order += [b for b in (NEG_INF,POS_INF,NEW) if b in cfg['woe_by_bin'] or
                  any(labels[s].eq(b).any() for s in segments)]
        order += [MISSING]
        train_stats={r['Bin']:r for r in cfg.get('bin_statistics',[])}
        for segment in segments:
            y=bundle['y'][segment]
            data=pd.DataFrame({'Bin':labels[segment],'y':y})
            observed=data.groupby('Bin').y.agg(N='size',Malos='sum').reindex(order,fill_value=0)
            observed['Buenos']=observed.N-observed.Malos
            low_count=0
            for position,label in enumerate(order):
                r=observed.loc[label];count=int(r.N)
                status='SIN_OBSERVACIONES' if count==0 else 'BAJO_VOLUMEN' if count<config['MIN_EVAL_BIN_N'] else 'SUFICIENTE'
                low_count+=int(status=='BAJO_VOLUMEN')
                tr=train_stats.get(label,{})
                def bound(name):
                    value=tr.get(name)
                    return np.nan if value is None else float(value)
                rows.append({'Variable':variable,'Segmento':segment,'Bin':label,'Orden_bin':position,
                    'Etiqueta':cfg['bin_labels'].get(label,label),'N':count,'Buenos':int(r.Buenos),'Malos':int(r.Malos),
                    'Dist_clientes':count/len(y) if len(y) else np.nan,
                    'Tasa_default':int(r.Malos)/count if count else np.nan,
                    'IV_train':cfg['iv_total'],'Estado_train':cfg['state'],
                    'WoE_train_congelado':cfg['woe_by_bin'].get(label,np.nan),
                    'Usa_prior_neutro':bool(count and label not in cfg['woe_by_bin']),
                    'Es_especial':label in SPECIAL,'Es_missing':label==MISSING,
                    'Limite_inferior':bound('Limite_inferior'),'Limite_superior':bound('Limite_superior'),
                    'Advertencia_volumen':status,'Volumen_insuficiente':status=='BAJO_VOLUMEN'})
            aggregates.append({'Variable':variable,'Segmento':segment,'N_observaciones':len(y),
                'Buenos':int(y.eq(0).sum()),'Malos':int(y.eq(1).sum()),
                'Tasa_default_global':float(y.mean()) if len(y) else np.nan,
                'Bins_con_datos':int(observed.N.gt(0).sum()),'Bins_vacios':int(observed.N.eq(0).sum()),
                'Bins_bajo_volumen':low_count,'N_missing':int(labels[segment].eq(MISSING).sum()),
                'N_categorias_nuevas':int(labels[segment].eq(NEW).sum()),
                'IV_train':cfg['iv_total'],'Estado_train':cfg['state']})
    columns=['Variable','Segmento','Bin','Orden_bin','Etiqueta','N','Buenos','Malos','Dist_clientes',
        'Tasa_default','IV_train','Estado_train','WoE_train_congelado','Usa_prior_neutro',
        'Es_especial','Es_missing','Limite_inferior','Limite_superior','Advertencia_volumen','Volumen_insuficiente']
    stats=pd.DataFrame(rows,columns=columns)
    comparison=[]
    for variable in variables:
        table=stats[stats.Variable.eq(variable)]
        for label,g in table.groupby('Bin',sort=False):
            by=g.set_index('Segmento')
            row={'Variable':variable,'Bin':label,'Orden_bin':int(g.Orden_bin.iloc[0]),'Etiqueta':g.Etiqueta.iloc[0]}
            for segment in segments:
                row.update({f'Clientes_{segment}':int(by.loc[segment,'N']),
                    f'Pct_clientes_{segment}':100*by.loc[segment,'Dist_clientes'],
                    f'Tasa_malos_{segment}':by.loc[segment,'Tasa_default'],
                    f'Advertencia_{segment}':by.loc[segment,'Advertencia_volumen']})
            row['Diferencia_pp_TEST_TRAIN']=100*(row['Tasa_malos_TEST']-row['Tasa_malos_TRAIN'])
            row['Diferencia_pp_OOT_TRAIN']=100*(row['Tasa_malos_OOT']-row['Tasa_malos_TRAIN'])
            row['Advertencia_bajo_volumen']='; '.join(s+':'+str(by.loc[s,'Advertencia_volumen']) for s in segments
                if by.loc[s,'Advertencia_volumen']!='SUFICIENTE')
            comparison.append(row)
    comparison_columns=['Variable','Bin','Orden_bin','Etiqueta']
    for segment in segments:
        comparison_columns += [f'Clientes_{segment}',f'Pct_clientes_{segment}',f'Tasa_malos_{segment}',f'Advertencia_{segment}']
    comparison_columns += ['Diferencia_pp_TEST_TRAIN','Diferencia_pp_OOT_TRAIN','Advertencia_bajo_volumen']
    return stats,pd.DataFrame(comparison,columns=comparison_columns),pd.DataFrame(aggregates,columns=[
        'Variable','Segmento','N_observaciones','Buenos','Malos','Tasa_default_global',
        'Bins_con_datos','Bins_vacios','Bins_bajo_volumen','N_missing','N_categorias_nuevas','IV_train','Estado_train'])


def sample_dashboard_limits(stats):
    limits={}
    for variable,g in stats.groupby('Variable',sort=False):
        rate=g.Tasa_default.max();volume=g.Dist_clientes.max()
        limits[variable]={'risk':(0.,max(.05,min(1.05,float(rate)*1.10))) if np.isfinite(rate) else (0.,.1),
                          'clients':(0.,max(.05,min(1.05,float(volume)*1.10))) if np.isfinite(volume) else (0.,1.)}
    return limits


def plot_sample_dashboard(stats,summary,artifact,segmento='TRAIN',columnas=4,por_pagina=12):
    """Same frozen variables/order/pages/scales in every observed sample."""
    if segmento not in ['TRAIN','TEST','OOT']:raise ValueError('Muestra desconocida')
    variables=artifact['approved_variables']
    ordered=summary.set_index('Variable').reindex(variables).reset_index()
    if not ordered.Estado_bivariado.eq('APROBADA').all():
        raise ValueError('Dashboard contiene variables no aprobadas en TRAIN')
    return plot_dashboard(stats[stats.Segmento.eq(segmento)],ordered,estado='APROBADA',
        columnas=columnas,por_pagina=por_pagina,segmento=segmento,limits=sample_dashboard_limits(stats))


def plot_variable(variable, selected_details, summary, ax=None,segmento='TRAIN',limits=None):
    stats = selected_details[selected_details.Variable.eq(variable)].copy()
    if stats.empty:
        row=summary.set_index('Variable').loc[variable]
        if row.Estado_bivariado!='NO_EVALUABLE':raise ValueError('Variable sin cortes seleccionados')
        if ax is None:figure,ax=plt.subplots(figsize=(12,4.5))
        else:figure=ax.figure
        secondary=ax.twinx()
        ax.text(.5,.5,'NO_EVALUABLE\n'+'\n'.join(textwrap.wrap(str(row.Motivos),45)),
                transform=ax.transAxes,ha='center',va='center',fontsize=8)
        ax.set_title(variable,fontsize=8);ax.set_axis_off();secondary.set_axis_off()
        return figure,ax,secondary
    ordinary = stats[~stats.Bin.isin(SPECIAL)].sort_values('Limite_inferior', kind='stable', na_position='last')
    quality = stats[stats.Bin.isin([NEG_INF, POS_INF, NEW])]
    missing = stats[stats.Bin.eq(MISSING)]
    stats = pd.concat([ordinary, quality, missing], ignore_index=True)
    if ax is None: figure, ax = plt.subplots(figsize=(12, 4.5))
    else: figure = ax.figure
    secondary = ax.twinx()
    x = np.arange(len(stats))
    ax.bar(x, stats.Dist_clientes, color='#4c87b6', alpha=.8)
    # Each special bin is an isolated point; no artificial connection to missing.
    if len(ordinary):
        secondary.plot(x[:len(ordinary)], stats.Tasa_default.iloc[:len(ordinary)], color='#c4483d', marker='o')
    for position in range(len(ordinary),len(stats)):
        secondary.plot([position], [stats.Tasa_default.iloc[position]], color='#c4483d', marker='D', linestyle='None')
    if len(stats)>len(ordinary):
        ax.axvline(len(ordinary)-.5, color='#666666', linestyle=':', linewidth=1)
        ax.axvspan(len(ordinary)-.5,len(stats)-.5,color='#ededed',alpha=.5,zorder=0)
    ax.set_xticks(x, ['\n'.join(textwrap.wrap('Ausencia de informacion' if code==MISSING else str(label), 22))
                     for code,label in zip(stats.Bin,stats.Etiqueta)], rotation=30, ha='right', fontsize=8)
    ax.yaxis.set_major_formatter(PercentFormatter(1)); secondary.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(bottom=0); secondary.set_ylim(bottom=0)
    if limits is not None and variable in limits:
        ax.set_ylim(*limits[variable]['clients']);secondary.set_ylim(*limits[variable]['risk'])
    ax.set_ylabel('% clientes'); secondary.set_ylabel('% malos', color='#c4483d')
    row = summary.set_index('Variable').loc[variable]
    ax.set_title('\n'.join(textwrap.wrap(variable, 45)) +
        f'\n{segmento} | IV TRAIN={row.IV_total:.4f} | bins={row.N_bins_efectivos} | fusiones={row.N_fusiones}' +
        f'\n{row.Monotonicidad} | {row.Estado_bivariado}', fontsize=8)
    if 'Advertencia_volumen' in stats:
        scarce=stats[stats.Advertencia_volumen.eq('BAJO_VOLUMEN')]
        if len(scarce):
            ax.text(.02,.97,'Bajo volumen: '+', '.join(f'{r.Bin} (n={r.N})' for r in scarce.itertuples()),
                transform=ax.transAxes,va='top',fontsize=6,color='#995500')
    return figure, ax, secondary


def plot_dashboard(selected_details, summary, estado='TODAS', columnas=4, por_pagina=12,segmento='TRAIN',limits=None):
    if not 1 <= por_pagina <= 12 or columnas < 1: raise ValueError('Columnas >=1; por_pagina 1..12')
    if estado not in ['TODAS', 'APROBADA', 'REQUIERE_REVISION','RECHAZADA','NO_EVALUABLE']: raise ValueError('Estado desconocido')
    variables = summary.Variable.tolist() if estado == 'TODAS' else summary.loc[summary.Estado_bivariado.eq(estado), 'Variable'].tolist()
    figures = []
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    for start in range(0, len(variables), por_pagina):
        page = variables[start:start+por_pagina]
        figure, axes = plt.subplots(math.ceil(len(page)/columnas), columnas,
            figsize=(6.5*columnas, 5.5*math.ceil(len(page)/columnas)), squeeze=False)
        for axis, variable in zip(axes.flat, page): plot_variable(variable, selected_details, summary, axis,segmento,limits)
        for axis in list(axes.flat)[len(page):]: axis.set_visible(False)
        figure.legend(handles=[Patch(color='#4c87b6', label='% clientes'),
            Line2D([], [], color='#c4483d', marker='o', label='% malos')],
            loc='upper center', ncol=2, bbox_to_anchor=(.5,.975), frameon=False)
        figure.suptitle(f'Bivariado {segmento} - {estado} - pagina {len(figures)+1}', y=.998)
        figure._biv_variables=list(page)
        figure._biv_segment=segmento
        figure.tight_layout(rect=(0,0,1,.94), h_pad=2.5, w_pad=3)
        figures.append(figure)
    return figures


def plot_monthly_rates(variable, stability, summary=None):
    data = stability[stability.Variable.eq(variable)]
    if data.empty:
        figure,axis=plt.subplots(figsize=(13,4.5))
        axis.text(.5,.5,'NO_EVALUABLE: sin estimaciones mensuales',ha='center',transform=axis.transAxes)
        axis.set_title(variable);axis.set_axis_off()
        return figure,axis
    calendar = pd.period_range(data.Periodo.min(), data.Periodo.max(), freq='M').astype(str)
    figure, axis = plt.subplots(figsize=(13,4.5))
    for label, group in data.groupby('Bin', sort=False):
        group = group.set_index('Periodo').reindex(calendar)
        values = group.Tasa_default.where(group.Estado_mes_bin.eq('EVALUABLE'))
        line, = axis.plot(calendar, values, marker='.', label=label)
        axis.fill_between(np.arange(len(calendar)), group.IC95_inferior.where(values.notna()).to_numpy(float),
            group.IC95_superior.where(values.notna()).to_numpy(float), color=line.get_color(), alpha=.08)
    axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.set_ylim(bottom=0)
    axis.set_title('\n'.join(textwrap.wrap(variable, 80)) + '\nTRAIN: tasa por bin; sombreado Wilson 95%; huecos no evaluables')
    axis.tick_params(axis='x',rotation=55); axis.legend(fontsize=7, ncol=4)
    figure.tight_layout()
    return figure, axis


def _remove_previous_generated_pngs(root, folders):
    """Called only after backup: remove our previous files, keep custom images."""
    managed = set()
    manifest = root/'evidencias/archivos_exportados.json'
    if manifest.exists():
        managed.update(json.loads(manifest.read_text(encoding='utf-8')))
    # B.1 filenames include rank; changed rankings otherwise leave stale copies.
    pointer = root/'evidencias/B2_base.json'
    if pointer.exists():
        baseline = Path(json.loads(pointer.read_text(encoding='utf-8'))['backup'])
        table = baseline/'reportes/df_resumen_bivariado.csv'
        if table.exists():
            for i, variable in enumerate(pd.read_csv(table).Variable,1):
                for folder in ['graficos_individuales','estabilidad']:
                    managed.add(f'{folder}/{i:03d}_{variable[:100]}.png')
    allowed = {(root/folder).resolve() for folder in folders}
    for name in managed:
        path = root/name
        if path.suffix.lower()=='.png' and path.resolve().parent in allowed and path.is_file():
            path.unlink()


def export_results(candidates, details, summary, stability, artifact, bundle,
                   transformed, transform_diagnostics, *, root=ROOT,
                   individual=True, dashboards=True, monthly_plots=True, comparison=None,
                   sample_stats=None,sample_comparison=None,sample_aggregate=None):
    """Exportacion explicita: no aprende ni selecciona candidatos."""
    root = Path(root)
    import shutil
    if any((root/folder).exists() for folder in ['reportes','graficos_individuales','subplots','estabilidad','matrices']):
        backup = root/'evidencias'/('respaldo_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        for folder in ['reportes','graficos_individuales','subplots','estabilidad','matrices']:
            if (root/folder).exists(): shutil.copytree(root/folder, backup/folder)
        if (root/'binning_woe_train.json').exists(): shutil.copy2(root/'binning_woe_train.json',backup/'binning_woe_train.json')
        requested = (['graficos_individuales'] if individual else [])
        requested += ['subplots'] if dashboards else []
        requested += ['estabilidad'] if monthly_plots else []
        _remove_previous_generated_pngs(root, requested)
    for folder in ['reportes','graficos_individuales','subplots','estabilidad','matrices','evidencias']:
        (root/folder).mkdir(parents=True,exist_ok=True)
    selected = details[details.Candidate_ID.isin(summary.Candidate_ID)].copy()
    tables={'df_candidatos_binning':candidates,'df_bins_detalle':details,
        'df_cortes_seleccionados':selected,'df_resumen_bivariado':summary,
        'df_estabilidad_tasas':stability,'df_diagnosticos_transformacion':transform_diagnostics}
    if comparison is not None:
        tables['comparacion_binning_antes_despues'] = comparison
    if sample_stats is not None:tables['df_estadisticas_bivariado_muestras']=sample_stats
    if sample_comparison is not None:tables['df_comparacion_bivariado_muestras']=sample_comparison
    if sample_aggregate is not None:tables['df_resumen_bivariado_muestras']=sample_aggregate
    if artifact.get('methodology_version')=='B.3':
        for key,name in [('approved_variables','variables_aprobadas_bivariado'),
                         ('rejected_variables','variables_rechazadas_bivariado'),
                         ('not_evaluable_variables','variables_no_evaluables_bivariado')]:
            tables[name]=pd.DataFrame({'Variable':artifact[key]})
    files=[]
    for name,table in tables.items():
        path=root/'reportes'/f'{name}.csv';table.to_csv(path,index=False);files.append(path)
    artifact=copy.deepcopy(artifact)
    artifact['input_version']=bundle['manifest']['version']
    artifact['input_manifest_sha256']=sha256(bundle['manifest_file'])
    path=save_binning(artifact,root/'binning_woe_train.json');files.append(path)
    restored=load_binning(path)
    for segment,X in bundle['X'].items():
        replay,_=transform_woe(X,restored)
        pd.testing.assert_frame_equal(replay,transformed[segment],check_exact=True)
        path=root/'matrices'/f'X_{segment.lower()}_woe.csv'
        transformed[segment].to_csv(path,index=True,index_label='_row_id');files.append(path)
    if individual:
        for i,variable in enumerate(summary.Variable,1):
            figure,_,_=plot_variable(variable,selected,summary)
            figure.tight_layout();path=root/'graficos_individuales'/f'{i:03d}_{variable[:100]}.png'
            figure.savefig(path,dpi=150);plt.close(figure);files.append(path)
    if dashboards:
        for page,figure in enumerate(plot_dashboard(selected,summary),1):
            path=root/'subplots'/f'bivariado_{page}.png';figure.savefig(path,dpi=140);plt.close(figure);files.append(path)
        if sample_stats is not None:
            for segment in ['TRAIN','TEST','OOT']:
                figures=plot_sample_dashboard(sample_stats,summary,artifact,segment)
                for page,figure in enumerate(figures,1):
                    path=root/'subplots'/f'{segment}_APROBADA_{page}.png'
                    figure.savefig(path,dpi=140);plt.close(figure);files.append(path)
    if monthly_plots:
        for i,variable in enumerate(summary.Variable,1):
            figure,_=plot_monthly_rates(variable,stability,summary)
            path=root/'estabilidad'/f'{i:03d}_{variable[:100]}.png'
            figure.savefig(path,dpi=140);plt.close(figure);files.append(path)
    (root/'evidencias/archivos_exportados.json').write_text(json.dumps({str(path.relative_to(root)):sha256(path) for path in files},indent=2),encoding='utf-8')
    return files


def _write_report_b2(root=ROOT):
    """Regenerate B.2 documentation using exported values, never fixed outcomes."""
    root=Path(root)
    summary=pd.read_csv(root/'reportes/df_resumen_bivariado.csv')
    candidates=pd.read_csv(root/'reportes/df_candidatos_binning.csv')
    comparison=pd.read_csv(root/'reportes/comparacion_binning_antes_despues.csv')
    artifact=load_binning(root/'binning_woe_train.json')
    old_review=comparison[comparison.Anteriormente_en_revision]
    remaining=summary[summary.Estado_bivariado.eq('REQUIERE_REVISION')]
    significant=comparison[comparison.Perdida_IV_relevante]
    approved=summary[summary.Estado_bivariado.eq('APROBADA')]
    tests_file=root/'evidencias/pruebas_B2.json'
    tests=json.loads(tests_file.read_text()) if tests_file.exists() else {}
    def table(frame):
        if frame.empty:return 'Sin registros.'
        clean=lambda v: str(v).replace('|','/').replace('\n',' ')
        return ('| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+
                '\n'.join('| '+' | '.join(clean(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)))
    cols=['Variable','Estado_anterior','Estado_nuevo','IV_anterior','IV_nuevo',
          'Bins_anteriores','Bins_finales','Numero_fusiones','Monotonicidad_nueva','Bin_minimo_final']
    lines=['# Bivariado WoE: mejora B.2', '',f'Generado: {now()}.', '',
        '## Resultado real', '',
        f'Variables: {len(summary)}. Configuraciones retenidas/evaluadas: {len(candidates)}; '+
        f"q originales: {int(candidates.Origen.eq('CUANTILES').sum())}. Estados: {summary.Estado_bivariado.value_counts().to_dict()}.",
        f'Monotonicidad ordinaria: {summary.Monotonicidad.value_counts().to_dict()}.',
        f'Minimo bin ordinario entre aprobadas: {approved.Min_pct_ordinario.min():.4%}.',
        f"Pruebas: {tests.get('tests','consultar evidencia')}; fallos: {tests.get('failures','N/D')}; errores: {tests.get('errors','N/D')}.", '',
        '## Variables que antes requerian revision', '',table(old_review[cols].round(6)), '',
        '## Variables que permanecen o pasan a revision', '',
        table(remaining[['Variable','IV_total','IV_ordinario','AUC_ordinario','Gap_tasas_ordinarias','Motivos']].round(6)), '',
        '## Perdida IV relevante', '',table(significant[['Variable','IV_anterior','IV_nuevo','Perdida_IV_pct']].round(6)), '',
        'La alerta compara con los cortes seleccionados B.1 y usa MAX_COMPARISON_IV_LOSS. No modifica la seleccion B.2. '+
        'Los candidatos nuevos parten de q=2..10; el numero de fusiones se cuenta desde su propio q inicial, no necesariamente desde los cortes B.1.', '',
        '## Cambios metodologicos y limites', '',
        '1. La funcion anterior ya excluia los nulos de la secuencia numerica. Se reforzo la identificacion por codigo __MISSING__ y el orden por limites/membresias reales, independientemente de etiquetas o del orden de filas. Se corrigio la visualizacion: missing al final, sombreado y con punto rojo independiente.',
        '2. MIN_BIN_PCT=0.05 usa el total TRAIN, incluidos los nulos, como denominador. Volumen y eventos minimos de aprobacion se exigen en bins ordinarios. Missing se mantiene separado y exento; si es escaso, se conserva una alerta de estimacion limitada, no una aprobacion de ese bin como estable.',
        '3. Se enumeran todas las particiones contiguas del binning de entrada y se comprueban sentidos creciente y decreciente. No es una cadena codiciosa de vecinos. Para cada numero de bins y sentido se retiene el mejor candidato por calidad e IV; todos los q originales se conservan. La busqueda es exhaustiva dentro de los limites/categorias iniciales de cada q, no sobre cualquier corte real imaginable.',
        '4. Los maximos iniciales q<=10 hacen finita la busqueda (a lo sumo 2^9 particiones por candidato, incluidas identidad/colapso). Se excluye el colapso a un bin. Cada fusion tiene historial de vecinos originales, conteos, tasa, WoE e IV antes/despues. No se promedian estimaciones.',
        '5. Discretas numericas de mas de dos valores pueden fusionar solo categorias consecutivas en su orden numerico y con mapa reproducible. Binarias y nominales mantienen sus categorias naturales; categorias pequenas se reportan y no se inventa un orden nominal ni una agrupacion de negocio.',
        '6. Una tasa plana es NO_EVALUABLE; la tendencia aproximada se distingue de la estricta. Las numericas no estrictas no se aprueban automaticamente. Las nominales carecen de orden: se informa NO_EVALUABLE de tendencia, con el resto de controles de calidad separado.',
        '7. Se consideran inicialmente >=3 bins. Dos bins requieren volumen/eventos, IV ordinario>=MIN_TWO_BIN_IV, gap>=MIN_TWO_BIN_RATE_GAP y AUC>=MIN_TWO_BIN_AUC; si hay alternativas viables >=3, la perdida IV no puede superar TWO_BIN_MAX_IV_LOSS. La regla preserva simplificacion con evidencia, no permite aprobar una variable con un solo bin.',
        '8. Los parametros de diferenciacion de riesgo e IV se declararon antes de calcular B.2. Son criterios operativos iniciales, no pruebas de significancia ni garantias predictivas. Se mantiene IV_RETAIN_FRACTION y se conserva toda variable no factible como REQUIERE_REVISION.',
        '9. WoE=ln(P(bin|bueno)/P(bin|malo)); IV=sum((P_bueno-P_malo)*WoE), sin escala porcentual. Se mantiene ALPHA=0.5 con normalizacion sobre todos los bins observados, incluido missing. Los bins puros conservan WoE finito. Al cambiar K, se renormalizan tambien las proporciones del bin missing aunque no se fusiono.',
        '10. Estabilidad mensual, tasas, WoE, AUC/Gini/KS y cortes se aprenden solo en TRAIN. AUC/KS/Gini usan -WoE y son metricas aparentes de desarrollo, optimistas al seleccionar multiples candidatos. Z mensual y Wilson son diagnosticos, no pruebas causales. TEST/OOT y nuevos clientes solo reciben transformaciones congeladas.',
        '11. Se mantiene la politica explicita de prior WoE neutral para bins/categorias no vistos TRAIN, con conteo de usos; puede bloquearse con UNSEEN_POLICY=raise. Esto no reemplaza por cero el WoE de un bin puro observado.',
        '12. Un IV alto no elimina variables ni demuestra fuga por si solo. Las variables de atraso requieren revisar definiciones/ventanas antes de modelar. Las aprobaciones se limitan a esta evaluacion de binning; no se entrenaron modelos ni se realizo seleccion multivariada.', '',
        '## Comparacion completa', '',table(comparison[cols].round(6)), '',
        '## Parametros B.2', '', '```json',json.dumps(artifact['parameters'],indent=2),'```', '',
        '## Artefactos y reproduccion', '',
        '- Notebook: ../../3.0.0.SCR_DES_ANLS_BIVARIADO.ipynb; mismas 15 secciones. BIV.8 muestra la comparacion completa y las siete revisiones previas; BIV.14 valida el 5%; BIV.15 exporta.',
        '- Modulo importado: ../../3.0.0.bivariado.py. Pruebas: ../../test_bivariado.py. Ejecutor: ../../3.0.1.ejecutar_bivariado.py.',
        '- Copia B.1 inmutable: ruta en evidencias/B2_base.json, con notebook, modulo, pruebas, reportes, matrices, graficos y JSON originales. No se sobrescribe la referencia de comparacion al reejecutar.',
        '- binning_woe_train.json: schema 2 / methodology_version B.2, parent_artifact_sha256, cortes anteriores y nuevos, historial, bins efectivos, missing y WoE finales. El loader acepta esquemas 1 y 2.',
        '- reportes/comparacion_binning_antes_despues.csv: las 29 variables reales y motivos de cada cambio; el numero no se fuerza si cambia el artefacto de entrada.',
        '- reportes/df_candidatos_binning.csv, df_bins_detalle.csv, df_cortes_seleccionados.csv, df_resumen_bivariado.csv y df_estabilidad_tasas.csv.',
        '- graficos_individuales/, subplots/ y estabilidad/: visualizaciones reales. Matrices WoE en matrices/. Originales inmutables en entradas/.',
        '- evidencias/pruebas_B2.txt y pruebas_B2.json; evidencias/validacion_B2.json y ejecucion.json.', '',
        '```powershell',
        '.\\.venv\\Scripts\\python.exe -m unittest test_bivariado -v',
        '.\\.venv\\Scripts\\python.exe 3.0.1.ejecutar_bivariado.py','```', '',
        'El ejecutor usa Python secuencial y captura plt.show; no se declara un kernel Jupyter interactivo. No se ejecutan notebooks anteriores, filtros ni PSI. Los hashes de fuentes, entradas v4 y reportes PSI se validan en la evidencia B.2. La serializacion JSON reproduce exactamente las matrices de los tres segmentos.']
    (root/'README.md').write_text('\n'.join(lines).replace(chr(92)*2,chr(92)),encoding='utf-8')
    return root/'README.md'


def active_parameters(config):
    inactive={'MIN_PREFERRED_BINS','MIN_IV_ORDINARY','MIN_TWO_BIN_IV','MIN_TWO_BIN_AUC',
              'MIN_TWO_BIN_RATE_GAP','TWO_BIN_MAX_IV_LOSS'} if config['POLICY_VERSION']=='B.3' else set()
    return {k:copy.deepcopy(v) for k,v in config.items() if k not in inactive}


def write_report(root=ROOT):
    root=Path(root)
    artifact=load_binning(root/'binning_woe_train.json')
    if artifact.get('methodology_version')!='B.3':return _write_report_b2(root)
    summary=pd.read_csv(root/'reportes/df_resumen_bivariado.csv')
    candidates=pd.read_csv(root/'reportes/df_candidatos_binning.csv')
    samples=pd.read_csv(root/'reportes/df_comparacion_bivariado_muestras.csv')
    aggregates=pd.read_csv(root/'reportes/df_resumen_bivariado_muestras.csv')
    comparison=pd.read_csv(root/'reportes/comparacion_binning_antes_despues.csv')
    tests_path=root/'evidencias/pruebas_B3.json'
    tests=json.loads(tests_path.read_text()) if tests_path.exists() else {}
    def table(frame):
        if frame.empty:return 'Sin registros.'
        clean=lambda v: str(v).replace('|','/').replace('\n',' ')
        return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+\
            '\n'.join('| '+' | '.join(map(clean,row))+' |' for row in frame.itertuples(index=False,name=None))
    counts=summary.Estado_bivariado.value_counts().reindex(['APROBADA','RECHAZADA','NO_EVALUABLE'],fill_value=0)
    focus=summary[summary.Variable.eq('variacion_calificacion_maxima_vs_actual_24m')]
    lines=['# Bivariado B.3: seleccion TRAIN y evaluacion congelada', '',f'Generado: {now()}.', '',
        f'Variables originales: {len(summary)}. Candidatos retenidos/evaluados: {len(candidates)}.',
        f'Estados: {counts.to_dict()}. Matrices WoE aprobadas: {len(artifact["approved_variables"])} columnas en las tres muestras.',
        f'Comparacion entre muestras: {len(samples)} filas variable/bin. Pruebas: {tests}.', '',
        '## Politica definitiva B.3', '',
        'APROBADA requiere simultaneamente >=2 bins ordinarios, IV TOTAL >=0.02 (incluido missing), monotonicidad numerica creciente/decreciente, cada bin ordinario >=5% del total TRAIN, minimos de observaciones/buenos/malos y diferenciacion de tasas >=MIN_RATE_GAP. Se conservan los controles estadisticos/mensuales B.2. Los cortes deben ser reproducibles y tecnicamente validos.',
        'Dos bins se juzgan con los mismos controles que tres o mas. No existe preferencia por >=3, filtro AUC adicional ni IV ordinario minimo adicional. El AUC/Gini/KS de TRAIN se mantiene como diagnostico aparente, no como condicion especial de dos bins.',
        'RECHAZADA significa incumplimiento conocido tras examinar alternativas: IV bajo, un unico bin, monotonicidad imposible, volumen/eventos o diferenciacion insuficientes. NO_EVALUABLE identifica TRAIN insuficiente, sin clases de target, variable sin observaciones ordinarias, o un error tecnico. Los errores se registran con IV faltante, nunca IV=0 artificial. Desalineacion de registros o FIT en segmentos distintos de TRAIN detiene la operacion.',
        'Se conserva la busqueda B.2 de particiones contiguas, ambos sentidos y recalculo desde conteos. Se retienen todos los q originales y las alternativas de la frontera. No se obliga a fusionar una configuracion ya valida. Se prefiere simplicidad entre candidatos validos dentro del margen IV_RETAIN_FRACTION.',
        'Missing conserva sus conteos, WoE e IV; esta exento de minimos ordinarios y excluido del orden/monotonicidad. Su estimacion escasa se advierte. Infinitos se mantienen como anomalias separadas. No se reinterpretan 0/-1/99/999.',
        'Las variables nominales mantienen categorias naturales y no reciben un orden numerico inventado; la tendencia no aplicable se informa por separado. En esta base las variables evaluadas son numericas.', '',
        '## Clasificacion real', '',table(summary[['Variable','IV_total','N_bins_efectivos','Monotonicidad','Estado_bivariado','Motivos']].round(6)), '',
        '## Variable solicitada', '',table(focus[['Variable','IV_total','N_bins_efectivos','AUC_ordinario','Estado_bivariado','Motivos']].round(6)), '',
        '## Comparacion B.2 / B.3', '',table(comparison[['Variable','Estado_anterior','Estado_nuevo','IV_anterior','IV_nuevo','Bins_finales','Motivo_del_cambio']].round(6)), '',
        '## Dashboards TRAIN / TEST / OOT', '',
        'TRAIN todas las variables se conserva. Los tres paneles de aprobadas utilizan exactamente la lista y orden aprendidos en TRAIN, iguales bins y ubicaciones por pagina, cuatro columnas y hasta doce variables. Las escalas de volumen/tasa son comunes por variable usando el rango observado entre las tres muestras; esto solo afecta la visualizacion.',
        'Cada panel emplea clientes y targets reales de su muestra. Los bins vacios permanecen con volumen 0 y tasa NaN; no se rellenan tasas con cero. Missing aparece al final con un punto rojo independiente y separacion visual. BAJO_VOLUMEN se advierte sin cambiar los cortes. Todos los buckets definidos en TRAIN se conservan, y categorias nuevas utilizan el bucket reservado documentado.',
        'evaluate_frozen_samples solo usa assign_bins y conteos observados. No llama learn_candidate, generate_candidates, search_adjacent_partitions ni select_configurations, y no estima nuevos WoE. Sus resultados no intervienen en la aprobacion.', '',
        '## Resumen por muestra', '',table(aggregates[['Variable','Segmento','N_observaciones','Malos','Tasa_default_global','Bins_vacios','Bins_bajo_volumen']].round(6)), '',
        'Las tasas se almacenan como proporciones; Pct_clientes_* y Diferencia_pp_* estan expresados en porcentajes/puntos porcentuales. Esta comparacion es descriptiva, no demuestra deterioro predictivo ni capacidad fuera de muestra.', '',
        '## Artefactos y ejecucion', '',
        '- binning_woe_train.json: schema 3, metodologia B.3, parent_artifact_sha256 B.2, cortes/WoE congelados y listas approved_variables/rejected_variables/not_evaluable_variables.',
        '- matrices/X_train_woe.csv, X_test_woe.csv, X_oot_woe.csv: solo aprobadas y en el mismo orden. transform_woe aplica esa seleccion por defecto en B.3. Las rechazadas conservan candidatos, metricas y cortes de auditoria en el JSON/reportes.',
        '- entradas/v4_VERSION/: las matrices originales completas y sus targets/ids permanecen intactos para XGBoost.',
        '- reportes/df_comparacion_bivariado_muestras.csv: clientes/porcentajes/tasas TRAIN, TEST y OOT, diferencias pp y advertencias por bin.',
        '- reportes/df_estadisticas_bivariado_muestras.csv y df_resumen_bivariado_muestras.csv: detalle y agregados observados.',
        '- subplots/bivariado_*.png: TRAIN todas; TRAIN_APROBADA_*.png, TEST_APROBADA_*.png y OOT_APROBADA_*.png: paneles comparables de aprobadas.',
        '- Copia previa B.2: evidencias/B3_base.json; conserva notebook, modulo, pruebas, reportes, matrices y JSON.',
        '- evidencias/pruebas_B3.txt, pruebas_B3.json, ejecucion.json y validacion_B3.json.', '',
        'Las 60 pruebas anteriores se conservan con fixtures explicitos POLICY_VERSION=B.2 para verificar compatibilidad. Los casos B.3 prueban la politica vigente por defecto y sus dashboards sin FIT. Los parametros B.2 inactivos se omiten a continuacion; no condicionan B.3.', '',
        '## Parametros activos', '', '```json',json.dumps(active_parameters(artifact['parameters']),indent=2),'```', '',
        '```powershell', '.\\.venv\\Scripts\\python.exe -m unittest test_bivariado -v',
        '.\\.venv\\Scripts\\python.exe 3.0.1.ejecutar_bivariado.py','```', '',
        'El notebook conserva BIV.1-BIV.15 y agrega paneles independientes BIV.12A-F despues de los graficos. BIV.15 exporta explicitamente. El ejecutor usa Python secuencial, no un kernel Jupyter interactivo. No se ejecutan el notebook 2.0, filtros ni PSI; sus artefactos se verifican por hash.']
    (root/'README.md').write_text('\n'.join(lines).replace(chr(92)*2,chr(92)),encoding='utf-8')
    return root/'README.md'


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['export-v4'])
    parser.parse_args()
    print(export_v4_from_psi())
