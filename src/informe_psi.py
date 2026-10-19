"""Build a report from explicitly exported tables; never calculate PSI."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import importlib.metadata
import json
import re
import shutil

import pandas as pd

ROOT = Path('4.ANALIS_UV_BV_MV/PSI_TEMPORAL')


def markdown_table(frame):
    if frame.empty:
        return 'Sin registros.'
    frame = frame.copy().fillna('—')
    clean = lambda value: str(value).replace('|', '\\|').replace('\n', ' ')
    header = '| ' + ' | '.join(map(clean, frame.columns)) + ' |'
    divider = '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'
    rows = ['| ' + ' | '.join(map(clean, row)) + ' |'
            for row in frame.itertuples(index=False, name=None)]
    return '\n'.join([header, divider] + rows)


def generate(root=None):
    root = ROOT if root is None else Path(root)
    reports = root / 'reportes'
    monthly = pd.read_csv(reports / 'df_psi_mensual.csv')
    summary = pd.read_csv(reports / 'df_resumen_psi.csv')
    references = pd.read_csv(reports / 'referencias.csv')
    consolidated = pd.read_csv(reports / 'df_reporte_univariado.csv')
    config_path = reports / 'parametros.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    evidence_path = root / 'evidencias/ejecucion.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8')) if evidence_path.exists() else {}
    amber, red = config.get('AMBER', .10), config.get('RED', .20)
    states = ['APROBADA', 'EN_OBSERVACION', 'REVISION_PRIORITARIA',
              'CANDIDATA_DESCARTE', 'NO_EVALUABLE']
    counts = pd.DataFrame({'Estado': states,
                           'Variables': [int(summary.Estado_PSI.eq(state).sum()) for state in states]})
    volumes = monthly[['Segmento', 'Periodo', 'N_observaciones']].drop_duplicates()
    segment_counts = volumes.groupby('Segmento').agg(
        Primer_periodo=('Periodo', 'min'), Ultimo_periodo=('Periodo', 'max'),
        Meses=('Periodo', 'nunique'), Registros=('N_observaciones', 'sum'),
        Min_registros_mes=('N_observaciones', 'min')).reset_index()
    exclusions = consolidated[consolidated.Decision.eq('Excluir')].groupby('Filtro').size()
    tops = monthly.nlargest(8, 'PSI')[['Variable', 'Segmento', 'Periodo', 'PSI']].round(6)
    critical_months = monthly.groupby(['Segmento', 'Periodo']).PSI.agg(['mean', 'max']).reset_index().nlargest(6, 'max').round(6)
    null_changes = monthly.assign(Delta_abs=monthly.Diferencia_nulos_pp.abs()).nlargest(5, 'Delta_abs')[[
        'Variable', 'Segmento', 'Periodo', 'Diferencia_nulos_pp', 'PSI_missing_component']].round(6)
    limited = monthly[monthly.Estado_calculo.ne('OK')]
    status_counts = monthly.Nivel_alerta.value_counts().rename_axis('Nivel_alerta').reset_index(name='Evaluaciones')
    smoothing = max((monthly.PSI - monthly.PSI_alpha_01).abs().max(),
                    (monthly.PSI - monthly.PSI_alpha_1).abs().max())
    reference_groups = references.groupby('Referencia_utilizada').agg(
        Variables=('Variable', 'nunique'), N_observaciones=('N_observaciones', 'first'),
        N_clientes=('N_clientes', 'first')).reset_index()
    tests_path = root / 'evidencias/pruebas_refactorizacion.txt'
    tests = tests_path.read_text(encoding='utf-8') if tests_path.exists() else ''
    match = re.search(r'Ran (\d+) tests?', tests)
    tests_message = (f'{match.group(1)} pruebas satisfactorias.'
                     if match and re.search(r'^OK$', tests, re.MULTILINE)
                     else 'Consulte las evidencias de pruebas disponibles; este informe no ejecuta pruebas.')
    images_by_section = {
        section: sum(item['images'] for item in evidence.get('cells', []) if item['section'] == section)
        for section in (7, 8, 9)
    }
    current_date = datetime.now(ZoneInfo('America/Lima')).strftime('%Y-%m-%d %H:%M %Z')
    text = f'''# PSI temporal: resultados y guía de ejecución

Informe generado: {current_date}. Todas las cifras siguientes proceden de los CSV exportados. Generar el informe no vuelve a calcular PSI.

## Flujo del notebook
PSI.1 configura; PSI.2 valida muestras; PSI.3 calcula una vez con export=False; PSI.4 muestra referencia; PSI.5 muestra el detalle mensual; PSI.6 clasifica; PSI.7 permite una variable opcional; **PSI.8 presenta el dashboard principal**; PSI.9 muestra heatmaps; PSI.10 consolida; PSI.11 construye v4; **PSI.12 exporta explícitamente**. Ninguna función gráfica necesita archivos PNG previos.

La reorganización conserva fórmula, nulos, suavizado, tratamiento de baja variabilidad, límites, referencia TRAIN y clasificación. Los filtros anteriores se mantienen. run_analysis(export=True) y export_plots() conservan compatibilidad con sus consumidores anteriores.

## Parámetros de la ejecución exportada

{markdown_table(pd.DataFrame(config.items(), columns=['Parámetro', 'Valor']))}

## Muestras y referencia reales

{markdown_table(segment_counts)}

Observaciones totales: **{int(segment_counts.Registros.sum())}**. Variables evaluadas: **{len(summary)}**. Evaluaciones mensuales: **{len(monthly)}**.

{markdown_table(reference_groups)}

Detalle por variable con nulos, método y bins efectivos: reportes/referencias.csv. Los meses incluidos en referencia no son evaluaciones temporalmente independientes. TEST/OOT no intervienen en el aprendizaje de bins ni en la clasificación.

## Clasificación y diagnósticos

{markdown_table(counts)}

{markdown_table(status_counts)}

Cobertura TRAIN mínima/máxima: {summary.Cobertura_temporal.min():.2%} / {summary.Cobertura_temporal.max():.2%}. Evaluaciones no válidas: {len(limited)}. Registros con infinitos contabilizados en las evaluaciones: {int(monthly.N_infinitos.sum())}.

Mayores PSI observados:

{markdown_table(tops)}

Meses con mayores máximos (indicadores descriptivos, no selección con TEST/OOT):

{markdown_table(critical_months)}

Cambios absolutos de nulos más grandes, expresados en puntos porcentuales:

{markdown_table(null_changes)}

Diferencia absoluta máxima al variar ALPHA a 0.1 o 1: {smoothing:.8f}. Las columnas de sensibilidad provienen del cálculo de PSI.3. La referencia full_train no se recalcula en el ejecutor; cualquier CSV histórico de sensibilidad se considera histórico y no se usa para concluir resultados de esta ejecución.

## Continuidad y filtros anteriores
Descartes originales por filtro: {exclusions.to_dict()}. Se reconstruye df_reporte_univariado sin duplicados. AUTO_DROP_PSI se mantiene en la configuración mostrada; las exclusiones físicas requieren una lista documentada. Los estados NO_EVALUABLE y CANDIDATA_DESCARTE conservan su necesidad de revisión aunque las variables continúen en v4.

PSI bajo no garantiza utilidad predictiva y PSI alto tampoco demuestra falta de capacidad predictiva. La decisión posterior debe combinar IV, tasas de default, colinealidad y validación de LR/XGBoost. Los filtros originales de concentración y nulos pueden excluir señales útiles; se preservan por alcance.

## Visualización y validación
En la última ejecución automatizada registrada, imágenes visibles en PSI.7: {images_by_section[7]}; páginas del dashboard PSI.8: {images_by_section[8]}; heatmaps PSI.9: {images_by_section[9]}. Los gráficos se construyen desde los dataframes en memoria. La visualización individual muestra una selección; no despliega todas las variables automáticamente.

{tests_message} Evidencia: evidencias/pruebas_refactorizacion.txt. Comparación antes/después: evidencias/refactor_validacion.json. Si se suministró --baseline, se exige igualdad exacta de los tres CSV después de la misma serialización. El ejecutor valida también un único cálculo PSI, ausencia de exportaciones antes de PSI.12, matrices v4, índices, target, segmentos e integridad de la base original.

Modo registrado: {evidence.get('execution', 'No hay ejecución automatizada registrada')}. Esta vía ejecuta celdas mediante Python y captura plt.show(); no se declara ejecución en kernel Jupyter interactivo ni se modifican protecciones del sistema. Los checks de matrices verifican su contrato de entrada a futuras etapas; no se ejecutan modelos posteriores inexistentes en el notebook.

## Qué ejecutar en PowerShell
Desde la raíz del proyecto:
```powershell
# Pruebas: no modifican el notebook ni los datos.
.\\.venv\\Scripts\\python.exe -m unittest test_psi_temporal -v
# Automatización opcional: guarda una copia ejecutada por defecto.
.\\.venv\\Scripts\\python.exe ejecutar_psi_notebook.py
# Para revisar todo sin exportar resultados PSI:
.\\.venv\\Scripts\\python.exe ejecutar_psi_notebook.py --skip-export
# Informe: necesita haber ejecutado PSI.12.
.\\.venv\\Scripts\\python.exe informe_psi.py
```
`psi_temporal.py` se importa; no se ejecuta directamente. Durante el trabajo interactivo basta ejecutar las celdas del notebook en orden. El ejecutor ofrece --in-place para guardar explícitamente sus salidas en el notebook principal, y --baseline RUTA para comparar con una copia previa.

## Archivos
- reportes/df_psi_mensual.csv, df_resumen_psi.csv, df_reporte_psi.csv, referencias.csv y df_reporte_univariado.csv.
- reportes/bins.json, distribuciones_bins.csv y parametros.json: auditoría aprendida en PSI.3 y exportada sin recalcular.
- subplots/ y heatmaps/: exportaciones solicitadas en PSI.12.
- graficos_individuales/: solo se generan si EXPORTAR_INDIVIDUALES_PSI=True; archivos de ejecuciones anteriores pueden permanecer como históricos.
- evidencias/inventario_refactorizacion.md: funciones, dependencias y responsabilidades anteriores.
- evidencias/refactor_base.json: ubicación de la copia de seguridad previa a esta refactorización.
- evidencias/ejecucion.json, ejecucion.txt y refactor_validacion.json: ejecución, salidas y comparación.

Las exportaciones guardan copias previas en evidencias/respaldo_FECHA. No se modifican fuente, target, segmentos ni filtros anteriores.
'''
    destination = root / 'README.md'
    if destination.exists():
        backup = root / 'evidencias' / ('README_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.md')
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, backup)
    destination.write_text(text, encoding='utf-8')
    versions = {name: importlib.metadata.version(name) for name in ['pandas', 'numpy', 'matplotlib', 'nbformat']}
    (root / 'evidencias').mkdir(parents=True, exist_ok=True)
    (root / 'evidencias/dependencias.json').write_text(json.dumps(versions, indent=2), encoding='utf-8')
    print(f'Informe generado: {destination}')
    return destination


if __name__ == '__main__':
    generate()
