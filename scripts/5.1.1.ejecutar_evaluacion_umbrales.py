"""Ejecuta el diagnóstico independiente 5.1 con cero ajustes y hashes previos."""
from pathlib import Path
from unittest.mock import patch
import base64
import contextlib
import importlib.util
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
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT=Path('5.1.EVALUACION_UMBRALES')
NOTEBOOK=Path('5.1.0.SCR_DES_EVALUACION_UMBRALES.ipynb')


def main():
    spec=importlib.util.spec_from_file_location('umb_runner','5.1.0.evaluacion_umbrales.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    evidence=ROOT/'evidencias';evidence.mkdir(parents=True,exist_ok=True)
    paths=[]
    for directory in ('4.ANALIS_UV_BV_MV','5.MODELO_LOGISTICO','2.OUTPUT'):
        paths.extend(p for p in Path(directory).rglob('*') if p.is_file())
    paths.extend(p for p in Path('.').glob('*.ipynb') if p!=NOTEBOOK)
    paths.extend(p for p in Path('.').glob('*.py') if not p.name.startswith('5.1.') and p.name!='test_evaluacion_umbrales.py')
    protected={str(p):module.sha256(p) for p in paths}
    (evidence/'hashes_archivos_previos.json').write_text(json.dumps(protected,indent=2),encoding='utf-8')
    fit_attempts=[]
    def forbidden_fit(*args,**kwargs):
        fit_attempts.append('AJUSTE_PROHIBIDO');raise AssertionError('No se permite entrenar/recalibrar el modelo 5.0')
    with patch.object(sm.Logit,'fit',forbidden_fit),patch.object(sm.Logit,'fit_regularized',forbidden_fit):
        tests=unittest.defaultTestLoader.loadTestsFromName('test_evaluacion_umbrales')
        stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(tests)
        (evidence/'pruebas.txt').write_text(stream.getvalue(),encoding='utf-8')
        tests_data=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors))
        (evidence/'pruebas.json').write_text(json.dumps(tests_data,indent=2),encoding='utf-8')
        if not result.wasSuccessful():raise RuntimeError(stream.getvalue())
        notebook=nbformat.read(NOTEBOOK,as_version=4);env={'__name__':'__main__'};current=None
        cells=[];logs=[];start=time.time();selection_calls=[]
        original_display,original_show=ipd.display,plt.show
        def display(*objects,**kwargs):
            for obj in objects:
                data={'text/plain':repr(obj)}
                if hasattr(obj,'_repr_html_'):
                    html=obj._repr_html_()
                    if html:data['text/html']=html
                current.outputs.append(nbformat.v4.new_output('display_data',data=data))
        def show(*args,**kwargs):
            for number in plt.get_fignums():
                fig=plt.figure(number);buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=110)
                current.outputs.append(nbformat.v4.new_output('display_data',data={
                    'image/png':base64.b64encode(buffer.getvalue()).decode(),'text/plain':repr(fig)}))
            plt.close('all')
        ipd.display,plt.show=display,show
        try:
            for cell in notebook.cells:
                if cell.cell_type!='code':continue
                current=cell;cell.outputs=[];section=cell.metadata['umb_section'];stream=io.StringIO()
                print(f'Ejecutando UMB.{section}',flush=True)
                with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                    exec(compile(cell.source,f'UMB_{section}','exec'),env)
                cell.execution_count=len(cells)+1
                if stream.getvalue():cell.outputs.insert(0,nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue()))
                cells.append(dict(section=section,images=sum('image/png' in o.get('data',{}) for o in cell.outputs)))
                logs.append(f'UMB.{section} OK\n'+stream.getvalue())
                if section==1:
                    original_choose=env['umb'].choose_candidates
                    def guarded_choose(grid,cfg):
                        assert current.metadata['umb_section']==6,'Selección posterior a evaluación heldout'
                        pd.testing.assert_frame_equal(grid,env['df_grilla_train_validacion'],check_exact=True)
                        selection_calls.append(dict(section=6,source='TRAIN_VALIDACION_INTERNA',N=len(env['train_validacion_interna'])))
                        return original_choose(grid,cfg)
                    env['umb'].choose_candidates=guarded_choose
            assert not fit_attempts and len(selection_calls)==1
            for file,digest in protected.items():assert module.sha256(file)==digest,f'Archivo previo alterado: {file}'
            pd.testing.assert_series_equal(env['modelo_congelado'].params,env['coeficientes_iniciales'],check_exact=True)
            assert module.sha256(env['CANDIDATES_PATH'])==env['hash_umbrales_congelados']
            assert len(env['df_grilla_umbrales'])==150
            for s in module.SEGMENTS:
                table=env['df_grilla_umbrales'].loc[env['df_grilla_umbrales'].Segmento.eq(s)]
                assert table.Recall.diff().dropna().le(0).all()
                assert table.Pct_clientes_alto_riesgo.diff().dropna().le(0).all()
            metrics5=pd.read_csv('5.MODELO_LOGISTICO/reportes/df_metricas_modelo.csv',float_precision='round_trip')
            np.testing.assert_allclose(env['df_calibracion_resumen'].Brier,metrics5.Brier,rtol=1e-12,atol=1e-12)
            candidates=env['df_umbrales_candidatos'];comparison=env['df_comparacion_umbrales'];distribution=env['df_distribucion_probabilidades']
            data=dict(execution='Python secuencial; tablas y figuras capturadas, sin kernel Jupyter interactivo',
                seconds=time.time()-start,methodology_version='5.1',fit_attempts=fit_attempts,
                selection_calls=selection_calls,internal_validation_N=len(env['train_validacion_interna']),
                limitation=module.LIMITATION,grid_N=50,segments_N=3,model_predictors=env['congelado']['features'],
                candidates=json.loads(candidates.to_json(orient='records',force_ascii=False)),
                comparison=json.loads(comparison.to_json(orient='records',force_ascii=False,double_precision=15)),
                probability_summary=json.loads(distribution.to_json(orient='records',double_precision=15)),
                protected_files_unchanged=len(protected),frozen_candidates_unchanged_after_holdout=True,
                tests=tests_data,cells=cells,source_sha256=module.sha256('5.1.0.evaluacion_umbrales.py'),
                runner_sha256=module.sha256(__file__))
            (evidence/'ejecucion.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
            def table(frame):
                clean=lambda v:str(v).replace('|','/').replace('\n',' ')
                return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(map(clean,row))+' |' for row in frame.itertuples(index=False,name=None))
            lines=['# Evaluación 5.1 — probabilidades y umbrales del modelo congelado','',
                'Nueve predictores y coeficientes intactos. Se leen el modelo y las probabilidades 5.0; se comprueba su reproducción sin entrenar.',
                module.LIMITATION,'',
                '## Distribución de probabilidades','',table(distribution.round(6)),'',
                '## Candidatos bajo escenarios ilustrativos','',table(candidates),'',
                f"Se usa el {env['CONFIG']['INTERNAL_VALIDATION_FRACTION']:.0%} de TRAIN estratificado, seed={env['CONFIG']['RANDOM_STATE']}; {len(env['train_validacion_interna'])} filas. El resto de TRAIN no elige candidatos.",
                'Los escenarios F1, recall objetivo y capacidad son configurables y no representan objetivos de negocio acordados. Desempates: mayor umbral. Se congela el JSON antes de evaluar heldout. No se declara un umbral operativo óptimo ni se estiman costos FP/FN no proporcionados.','',
                '## Evaluación de candidatos congelados','',table(comparison[['Candidato','Segmento','Umbral','Precision','Recall','F1','Accuracy','Pct_clientes_alto_riesgo']].round(6)),'',
                'Las tablas y curvas completas TEST/OOT son diagnósticas; no se usan para cambiar candidatos. La regla de clasificación es probabilidad >= umbral. Métricas indefinidas se guardan NaN con la causa.','',
                '## Deciles y calibración','',
                'Cada muestra se ordena por probabilidad descendente y _row_id ascendente, sin usar el target para ordenar. Decil 1 es el mayor riesgo. Los tamaños difieren como máximo en una fila. Un empate de probabilidad puede dividirse por el identificador; se marca en el reporte y no significa una diferencia real de riesgo. Los deciles son descriptivos, no nuevos cortes WoE ni umbrales seleccionados.',
                table(env['df_deciles_riesgo'].loc[env['df_deciles_riesgo'].Decil.eq(1),['Segmento','N','Tasa_default','Captura_defaults_acumulada','Lift_decil']].round(6)),'',
                table(env['df_calibracion_resumen'].round(6)),'',
                'Brier no depende del umbral y coincide con fase 5.0. La referencia constante usa la prevalencia TRAIN. ECE usa diez intervalos fijos de probabilidad, es descriptivo y depende del binning. Bins vacíos conservan N=0 y tasas/promedios ausentes. No se ajusta ningún calibrador.','',
                '## Artefactos y reproducción','',
                '- Notebook ../5.1.0.SCR_DES_EVALUACION_UMBRALES.ipynb; módulo ../5.1.0.evaluacion_umbrales.py.',
                '- umbrales_candidatos_congelados.json: criterios, configuración, umbrales, hashes de entradas y partición interna.',
                '- reportes/: distribución, grilla interna TRAIN, candidatos, comparación de muestras, grilla completa (150 filas), deciles/asignaciones y calibración.',
                '- graficos/: distribución, Precision/Recall vs umbral, captura, default por decil, lift y calibración.',
                '- evidencias/: pruebas, ejecución por celda, manifiesto de partición y hashes de archivos previos.',
                f'Pruebas: {tests_data}; archivos previos intactos: {len(protected)}; ajustes realizados: 0.','',
                '```powershell','.\\.venv\\Scripts\\python.exe 5.1.1.ejecutar_evaluacion_umbrales.py','```','',
                'El ejecutor utiliza Python secuencial y captura display/plt.show; no declara ejecución por kernel Jupyter. No modifica los artefactos 5.0, predicciones ni etapas anteriores. No se crean commits.']
            (ROOT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
            print(candidates.to_string(index=False));print(comparison[['Candidato','Segmento','Umbral','Precision','Recall','F1','Pct_clientes_alto_riesgo']].to_string(index=False))
            print(f'Pruebas: {tests_data}; archivos previos intactos: {len(protected)}; fit: {len(fit_attempts)}')
        except Exception:
            logs.append(traceback.format_exc());(evidence/'error_ejecucion.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
        finally:
            ipd.display,plt.show=original_display,original_show
            nbformat.write(notebook,NOTEBOOK);(evidence/'ejecucion.txt').write_text('\n'.join(logs),encoding='utf-8')
        (evidence/'notebook_sha256.txt').write_text(module.sha256(NOTEBOOK),encoding='utf-8')


if __name__=='__main__':main()
