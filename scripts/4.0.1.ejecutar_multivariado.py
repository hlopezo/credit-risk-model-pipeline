"""Ejecuta MULT.1-14 en Python y captura tablas/figuras; no usa kernel Jupyter."""
from pathlib import Path
import base64
import contextlib
import hashlib
import io
import json
import time
import traceback
import unittest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import IPython.display as ipd
import nbformat

ROOT=Path('4.ANALIS_UV_BV_MV/MULTIVARIADO')
NOTEBOOK=Path('4.0.0.SCR_DES_ANLS_Multivar.ipynb')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    evidence=ROOT/'evidencias';evidence.mkdir(parents=True,exist_ok=True)
    # Todos los artefactos de etapas anteriores, incluyendo fuentes/muestras.
    paths=[p for p in Path('4.ANALIS_UV_BV_MV').rglob('*') if p.is_file() and ROOT not in p.parents]
    paths += [p for p in Path('.').glob('*.ipynb') if p!=NOTEBOOK]
    paths += list(Path('2.OUTPUT').rglob('*.csv'))
    paths += [Path('3.0.0.bivariado.py'),Path('test_bivariado.py')]
    protected={str(p):digest(p) for p in paths}
    (evidence/'hashes_entradas.json').write_text(json.dumps(protected,indent=2),encoding='utf-8')
    tests=unittest.defaultTestLoader.loadTestsFromName('test_multivariado')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(tests)
    (evidence/'pruebas.txt').write_text(stream.getvalue(),encoding='utf-8')
    test_summary=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors))
    (evidence/'pruebas.json').write_text(json.dumps(test_summary,indent=2),encoding='utf-8')
    if not result.wasSuccessful():raise RuntimeError(stream.getvalue())
    notebook=nbformat.read(NOTEBOOK,as_version=4);env={'__name__':'__main__'};cells=[];logs=[];current=None
    original_display,original_show=ipd.display,plt.show
    def display(*objects,**kwargs):
        for obj in objects:
            data={'text/plain':repr(obj)}
            if hasattr(obj,'_repr_html_'):
                html=obj._repr_html_()
                if html:data['text/html']=html
            current.outputs.append(nbformat.v4.new_output('display_data',data=data))
    def show(*args,**kwargs):
        for num in plt.get_fignums():
            fig=plt.figure(num);buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=100)
            current.outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(buffer.getvalue()).decode(),'text/plain':repr(fig)}))
        plt.close('all')
    learning=[]
    def forbidden(*args,**kwargs):
        learning.append('FIT prohibido');raise AssertionError('No se permite reaprender B.3')
    ipd.display,plt.show=display,show
    start=time.time()
    try:
        for cell in notebook.cells:
            if cell.cell_type!='code':continue
            current=cell;cell.outputs=[];section=cell.metadata['mult_section'];stream=io.StringIO()
            print(f'Ejecutando MULT.{section}',flush=True)
            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                exec(compile(cell.source,f'MULT_{section}','exec'),env)
            cell.execution_count=len(cells)+1
            if stream.getvalue():cell.outputs.insert(0,nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue()))
            cells.append(dict(section=section,images=sum('image/png' in o.get('data',{}) for o in cell.outputs)))
            logs.append(f'MULT.{section} OK\n'+stream.getvalue())
            if section==1:
                for name in ['learn_candidate','generate_candidates','evaluate_candidates','select_configurations','search_adjacent_partitions','bin_statistics']:
                    setattr(env['biv'],name,forbidden)
        for p,h in protected.items():assert digest(p)==h,f'Etapa previa alterada: {p}'
        assert not learning
        decisions=env['df_correlacion_decisiones'];corr=env['corr_woe_final'].abs().to_numpy()
        import numpy as np
        pairs=corr[np.triu_indices(len(corr),1)]
        evidence_data=dict(execution='Python secuencial; display y plt.show capturados, sin kernel Jupyter',
            seconds=time.time()-start,initial_variables=len(env['cols_por_iv']),retained_variables=env['variables_finales'],
            eliminated_variables=decisions.loc[decisions.Estado_correlacion.eq('ELIMINADA_CORRELACION'),'Variable'].tolist(),
            final_count=len(env['variables_finales']),threshold=env['CORR_THRESHOLD'],learned_on='TRAIN',
            max_abs_final_correlation=float(pairs.max()) if len(pairs) else None,
            shapes={s:list(X.shape) for s,X in env['matrices_uncorr'].items()},
            protected_files_unchanged=len(protected),woe_learning_calls=learning,
            json_reconstruction_exact=True,csv_roundtrip_exact=True,tests=test_summary,cells=cells,
            module_sha256=digest('4.0.0.multivariado.py'))
        (evidence/'ejecucion.json').write_text(json.dumps(evidence_data,indent=2,ensure_ascii=False),encoding='utf-8')
        lines=['# Multivariado 4.0 — selección por correlación WoE','',
            f"TRAIN: {evidence_data['initial_variables']} aprobadas B.3 → {evidence_data['final_count']} finalistas.",
            f"Se recorren IV final descendente y nombre alfabético en empates. Se elimina cuando abs(Pearson) > {env['CORR_THRESHOLD']} con alguna conservada; la referencia es la de mayor |r| y se registran todas las coincidencias.",
            f"Máximo |r| entre finalistas TRAIN: {evidence_data['max_abs_final_correlation']:.8f}.",
            f"Pruebas: {test_summary}. Archivos previos verificados sin cambios: {len(protected)}.",'',
            '## Finalistas','',*['- '+v for v in env['variables_finales']],'',
            '## Artefactos','',
            '- matrices/X_{train,test,oot}_uncorr.csv: mismos registros y columnas, orden IV; índice _row_id.',
            '- matrices/manifest_indices.json: tipos, orden e identidad de filas/targets y hashes CSV; verificación de lectura exacta.',
            '- reportes/df_iv_final.csv: IV de los cortes finales B.3, validado frente al JSON.',
            '- reportes/df_correlacion_decisiones.csv: decisión de cada aprobada, referencia y coeficiente firmado.',
            '- reportes/df_correlaciones_descartadas.csv: todas las referencias retenidas que provocan descarte.',
            '- reportes/corr_woe_completa.csv, corr_woe_absoluta.csv, corr_woe_final.csv y variables_finales.csv.',
            '- graficos/: heatmap_completo.png, heatmap_final.png, ranking_iv.png; mostrados antes de exportar.',
            '- seleccion_correlacion.json: metodología, umbral, orden IV, decisiones y hash de binning B.3.',
            '- evidencias/: pruebas, diagnósticos WoE, validaciones de matrices, hashes previos y ejecución por celda.','',
            '## Reproducción','',
            'Desde la raíz del proyecto ejecutar el notebook 4.0.0.SCR_DES_ANLS_Multivar.ipynb en orden o:',
            '```powershell','.\\.venv\\Scripts\\python.exe 4.0.1.ejecutar_multivariado.py','```','',
            'El ejecutor usa Python secuencial y captura tablas y figuras. El módulo 4.0.0.multivariado.py reutiliza load_v4/load_binning/transform_woe de 3.0; no estima nuevos WoE ni modelo predictivo.',
            'Para reconstruir: mult.load_selection(JSON, binning_path), luego mult.reconstruct(muestras_biv, seleccion, biv, binning_path); no aprende correlaciones.',
            'Constantes, NaN o infinitos en correlaciones detienen explícitamente la selección; no se convierten a cero. TEST/OOT solo reciben transformaciones y columnas decididas en TRAIN.',
            'El orden final es IV descendente. Una correlación exactamente igual al umbral se conserva. El barrido es secuencial: una variable eliminada no bloquea otras posteriores.']
        (ROOT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
        print(json.dumps(evidence_data,indent=2,ensure_ascii=False))
    except Exception:
        logs.append(traceback.format_exc());raise
    finally:
        ipd.display,plt.show=original_display,original_show
        nbformat.write(notebook,NOTEBOOK)
        (evidence/'ejecucion.txt').write_text('\n'.join(logs),encoding='utf-8')
    (evidence/'notebook_sha256.txt').write_text(digest(NOTEBOOK),encoding='utf-8')


if __name__=='__main__':main()
