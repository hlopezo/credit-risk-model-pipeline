"""Run notebook cells sequentially in Python; capture tables and plt.show figures.

Default writes an executed copy. --in-place explicitly updates the main notebook.
This runner does not start an interactive Jupyter kernel or install dependencies.
"""
from pathlib import Path
import argparse
import base64
import contextlib
from datetime import datetime
import hashlib
import io
import json
import math
import re
import shutil
import sys
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import nbformat
import IPython.display as ipd
import pandas as pd
import psi_temporal as psi

NOTEBOOK = Path('2.0.0.SCR_DES_ANLS_UNV_BIV_MULT.ipynb')


def output_fingerprint():
    """Track exported results, excluding logs/backups produced by this runner."""
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ['reportes', 'graficos_individuales', 'subplots', 'heatmaps']
        for path in sorted((psi.ROOT / folder).glob('*')) if path.is_file()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-place', action='store_true', help='Guardar salidas en el notebook principal')
    parser.add_argument('--output', type=Path, help='Ruta de la copia ejecutada')
    parser.add_argument('--skip-export', action='store_true', help='Omitir únicamente PSI.12')
    parser.add_argument('--baseline', type=Path, help='Comparar tres CSV contra una copia previa')
    args = parser.parse_args(argv)
    if args.in_place and args.output:
        parser.error('Seleccione --in-place o --output')
    root = psi.ROOT
    evidence_dir = root / 'evidencias'
    evidence_dir.mkdir(parents=True, exist_ok=True)
    output = NOTEBOOK if args.in_place else (args.output or evidence_dir / 'notebook_ejecutado.ipynb')
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = Path('2.OUTPUT/1.1.0.TAB_SEGM_TRAIN_TEST_OOT.csv')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    env = {'__name__': '__main__', 'PSI_EJECUCION_AUTOMATICA': True}
    logs, cell_evidence = [], []
    current = None
    section = None
    calculation_calls = []
    original_display, original_show, original_run = ipd.display, plt.show, psi.run_analysis
    initial_exports = None

    def append_figure(figure):
        buffer = io.BytesIO()
        figure.savefig(buffer, format='png', dpi=110)
        current.outputs.append(nbformat.v4.new_output('display_data', data={
            'text/plain': repr(figure),
            'image/png': base64.b64encode(buffer.getvalue()).decode('ascii'),
        }))

    def capture(*objects, **kwargs):
        for obj in objects:
            if isinstance(obj, Figure):
                append_figure(obj)
                continue
            data = {'text/plain': repr(obj)}
            if isinstance(obj, ipd.Image):
                data['image/png'] = base64.b64encode(obj.data).decode('ascii')
            elif hasattr(obj, '_repr_html_'):
                html = obj._repr_html_()
                if html:
                    data['text/html'] = html
            current.outputs.append(nbformat.v4.new_output('display_data', data=data))

    def show(*args, **kwargs):
        # Match the inline backend: display all open figures, then close them.
        for number in plt.get_fignums():
            append_figure(plt.figure(number))
        plt.close('all')

    def counted_run(*args, **kwargs):
        calculation_calls.append({'section': section,
                                  'reference_mode': kwargs.get('reference_mode', psi.REFERENCE_MODE)})
        return original_run(*args, **kwargs)

    ipd.display, plt.show, psi.run_analysis = capture, show, counted_run
    env['display'] = capture
    try:
        count = 0
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == 'markdown':
                match = re.match(r'## PSI\.(\d+)\b', cell.source)
                if match:
                    section = int(match.group(1))
                continue
            current = cell
            cell.outputs = []
            if section == 1 and initial_exports is None:
                initial_exports = output_fingerprint()
            if section == 12 and args.skip_export:
                cell.execution_count = None
                logs.append(f'CELL {index} PSI.12 SKIPPED (--skip-export)')
                continue
            text_output = io.StringIO()
            print(f'Executing cell {index}' + (f' PSI.{section}' if section else ''), flush=True)
            with contextlib.redirect_stdout(text_output), contextlib.redirect_stderr(text_output):
                exec(compile(cell.source, f'notebook_cell_{index}', 'exec'), env)
            count += 1
            cell.execution_count = count
            if text_output.getvalue():
                cell.outputs.insert(0, nbformat.v4.new_output(
                    'stream', name='stdout', text=text_output.getvalue()))
            images = sum('image/png' in item.get('data', {}) for item in cell.outputs)
            cell_evidence.append({'index': index, 'section': section,
                                  'images': images, 'outputs': len(cell.outputs)})
            logs.append(f'CELL {index} PSI.{section} OK; images={images}\n' + text_output.getvalue())
            if section == 11:
                assert output_fingerprint() == initial_exports, 'Escritura de resultados antes de PSI.12'
            plt.close('all')

        for segment in ['train', 'test', 'oot']:
            original = env[f'X_{segment}_v3']
            final = env[f'X_{segment}_v4']
            pd.testing.assert_frame_equal(final, original.drop(columns=env['variables_descartar_psi']))
            assert final.index.equals(env[f'y_{segment}'].index)
        data, summary = env['df_psi_mensual'], env['df_resumen_psi']
        assert not data.duplicated(['Variable', 'Segmento', 'Periodo']).any()
        assert not env['df_reporte_univariado'].duplicated(['Variable', 'Filtro']).any()
        original_source = pd.read_csv(source)
        pd.testing.assert_frame_equal(env['df'], original_source)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        assert len(calculation_calls) == 1 and calculation_calls[0]['section'] == 3
        assert sum(item['images'] for item in cell_evidence if item['section'] == 7) == 1
        assert sum(item['images'] for item in cell_evidence if item['section'] == 8) == len(env['figuras_dashboard_psi'])
        assert sum(item['images'] for item in cell_evidence if item['section'] == 9) == 3
        comparisons = {}
        if args.baseline:
            tables = {'df_psi_mensual.csv': data, 'df_resumen_psi.csv': summary,
                      'referencias.csv': env['df_referencias_psi']}
            for name, table in tables.items():
                # CSV roundtrip matches the original serialization, including float parsing.
                current_csv = pd.read_csv(io.StringIO(table.to_csv(index=False)))
                baseline = pd.read_csv(args.baseline / name)
                pd.testing.assert_frame_equal(current_csv, baseline, check_exact=True)
                comparisons[name] = {'equal_exact': True, 'rows': len(table),
                                     'columns': len(table.columns)}
        evidence = {
            'python': sys.version, 'executable': sys.executable,
            'source_sha256': source_hash, 'rows': len(env['df']),
            'v3_columns': env['X_train_v3'].shape[1], 'v4_columns': env['X_train_v4'].shape[1],
            'v4_preserved': not env['variables_descartar_psi'], 'alignment_preserved': True,
            'target_segments_source_unchanged': True, 'monthly_no_duplicates': True,
            'execution': 'Python secuencial en proceso; no kernel Jupyter interactivo',
            'calculation_calls': calculation_calls, 'no_exports_before_PSI12': True,
            'export_executed': not args.skip_export, 'notebook_output': str(output),
            'states': summary.Estado_PSI.value_counts().to_dict(),
            'previous_variability_excluded': env['variables_descartadas'],
            'previous_missing_excluded': env['variables_descartadas_f3'],
            'initial_predictors': env['X_train'].shape[1],
            'configuration': env['configuracion_psi'], 'cells': cell_evidence,
            'baseline_comparisons': comparisons,
        }
        evidence_path = evidence_dir / ('ejecucion_sin_exportar.json' if args.skip_export else 'ejecucion.json')
        evidence_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding='utf-8')
        if args.baseline:
            (evidence_dir / 'refactor_validacion.json').write_text(
                json.dumps(evidence, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({key: evidence[key] for key in ['states', 'calculation_calls', 'baseline_comparisons']}, indent=2))
    except Exception:
        logs.append(traceback.format_exc())
        raise
    finally:
        ipd.display, plt.show, psi.run_analysis = original_display, original_show, original_run
        if output == NOTEBOOK:
            backup = evidence_dir / ('notebook_antes_ejecucion_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.ipynb')
            shutil.copy2(NOTEBOOK, backup)
        output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output)
        log_path = evidence_dir / ('ejecucion_sin_exportar.txt' if args.skip_export else 'ejecucion.txt')
        log_path.write_text('\n'.join(logs), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
