"""Ejecutor secuencial MOD.1-20 con evidencia, pruebas y captura de tablas/figuras."""
from pathlib import Path
import base64
import contextlib
import io
import json
import platform
import sys
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

ROOT=Path('5.MODELO_LOGISTICO')
NOTEBOOK=Path('5.0.0.SCR_DES_MODELO_LOGISTICO.ipynb')


def main():
    import importlib.util
    spec=importlib.util.spec_from_file_location('modelo_runner','5.0.0.modelo_logistico.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    evidence=ROOT/'evidencias';evidence.mkdir(parents=True,exist_ok=True)
    paths=[p for p in Path('4.ANALIS_UV_BV_MV').rglob('*') if p.is_file()]
    paths += [p for p in Path('.').glob('*.ipynb') if p!=NOTEBOOK]
    paths += list(Path('2.OUTPUT').rglob('*.csv'))
    paths += [p for p in Path('.').glob('*.py') if not p.name.startswith('5.0.') and p.name!='test_modelo_logistico.py']
    protected={str(p):module.sha256(p) for p in paths}
    (evidence/'hashes_etapas_previas.json').write_text(json.dumps(protected,indent=2),encoding='utf-8')
    tests=unittest.defaultTestLoader.loadTestsFromName('test_modelo_logistico')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(tests)
    (evidence/'pruebas.txt').write_text(stream.getvalue(),encoding='utf-8')
    tests_data=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors))
    (evidence/'pruebas.json').write_text(json.dumps(tests_data,indent=2),encoding='utf-8')
    if not result.wasSuccessful():raise RuntimeError(stream.getvalue())
    notebook=nbformat.read(NOTEBOOK,as_version=4);env={'__name__':'__main__'}
    current=None;logs=[];cells=[];fit_calls=[];forbidden_calls=[];start=time.time()
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
            figure=plt.figure(num);buffer=io.BytesIO();figure.savefig(buffer,format='png',dpi=110)
            current.outputs.append(nbformat.v4.new_output('display_data',data={
                'image/png':base64.b64encode(buffer.getvalue()).decode(),'text/plain':repr(figure)}))
        plt.close('all')
    def forbidden(*args,**kwargs):
        forbidden_calls.append(current.metadata['mod_section'])
        raise AssertionError('No se permite recalcular WoE/IV/cortes en 5.0')
    ipd.display,plt.show=display,show
    try:
        for cell in notebook.cells:
            if cell.cell_type!='code':continue
            current=cell;current.outputs=[];section=cell.metadata['mod_section'];stream=io.StringIO()
            print(f'Ejecutando MOD.{section}',flush=True)
            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                exec(compile(cell.source,f'MOD_{section}','exec'),env)
            cell.execution_count=len(cells)+1
            if stream.getvalue():cell.outputs.insert(0,nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue()))
            cells.append(dict(section=section,images=sum('image/png' in o.get('data',{}) for o in cell.outputs)))
            logs.append(f'MOD.{section} OK\n'+stream.getvalue())
            if section==1:
                mod=env['mod'];original_import=mod.import_bivariado;original_fit=mod.fit_logit
                def guarded_import():
                    biv=original_import()
                    for name in ['transform_woe','learn_candidate','generate_candidates','evaluate_candidates','select_configurations','search_adjacent_partitions','bin_statistics']:
                        setattr(biv,name,forbidden)
                    return biv
                def guarded_fit(X,y,config):
                    pd.testing.assert_frame_equal(X,env['X_train_modelo_inicial'].loc[:,X.columns],check_exact=True)
                    pd.testing.assert_series_equal(y,env['y_train'],check_exact=True)
                    fit_calls.append(dict(section=current.metadata['mod_section'],n_rows=len(X),columns=X.columns.tolist()))
                    return original_fit(X,y,config)
                mod.import_bivariado=guarded_import;mod.fit_logit=guarded_fit
            if isinstance(section,int) and section>=10:
                assert len(fit_calls)==len(env['estado_seleccion'].history),'Ajuste adicional innecesario'
        state=env['estado_seleccion'];inputs=env['entradas_modelo'];config=env['CONFIG']
        assert state.status=='VALIDADO' and not forbidden_calls
        assert env['modelo_sm'] is state.model
        assert all(call['section'] in (5,6,7,8) for call in fit_calls)
        assert (state.model.params.drop('const')*state.expected_sign>=0).all()
        assert state.model.pvalues.drop('const').le(config['P_VALUE_THRESHOLD']).all()
        assert state.shap['table'].SHAP_pct.ge(config['SHAP_MIN_PCT']).all()
        for p,h in protected.items():assert module.sha256(p)==h,f'Archivo previo alterado: {p}'
        for s in module.SEGMENTS:
            X=env['matrices_modelo'][s]
            restored=pd.read_csv(ROOT/f'matrices/X_{s.lower()}_modelo.csv',index_col='_row_id',float_precision='round_trip')
            restored.index.name=X.index.name;pd.testing.assert_frame_equal(X,restored,check_exact=True)
            pred=pd.read_csv(ROOT/f'predicciones/predicciones_{s.lower()}.csv',index_col='_row_id',float_precision='round_trip')
            assert pred.index.equals(inputs['y'][s].index)
            np.testing.assert_array_equal(pred.Target,inputs['y'][s])
            np.testing.assert_allclose(pred.Probabilidad_default,env['predicciones_modelo'][s],rtol=1e-12,atol=1e-12)
        history=state.df_iteraciones_modelo;metrics=env['df_metricas_modelo'];iv=env['df_filtro_iv']
        data=dict(execution='Python secuencial; tablas y plt.show capturados, sin kernel Jupyter interactivo',
            methodology_version='5.0',seconds=time.time()-start,python=sys.version,platform=platform.platform(),
            versions=dict(statsmodels=module.sm.__version__,shap=module.shap.__version__,pandas=pd.__version__,numpy=np.__version__),
            convention=env['convencion_woe'],variables_received=env['variables_recibidas'],
            excluded_iv=iv.loc[iv.Estado_IV.eq('EXCLUIDA_IV'),'Variable'].tolist(),
            eliminated={c:history.loc[history.Criterio_eliminacion.eq(c),'Variable_eliminada'].tolist() for c in ('SIGNO','P_VALUE','SHAP')},
            variables_final=state.variables,number_final=len(state.variables),converged=bool(state.model.mle_retvals['converged']),
            shapes={s:list(X.shape) for s,X in env['matrices_modelo'].items()},
            fit_calls=fit_calls,forbidden_prior_calculation_calls=forbidden_calls,
            protected_files_unchanged=len(protected),loaded_model_predictions_match=True,
            probability_tolerance=dict(rtol=1e-12,atol=1e-12),csv_matrices_roundtrip_exact=True,
            metrics=json.loads(metrics.to_json(orient='records',double_precision=15)),
            deltas=json.loads(env['df_deltas_modelo'].to_json(orient='records',double_precision=15)),tests=tests_data,cells=cells,
            module_sha256=module.sha256('5.0.0.modelo_logistico.py'),runner_sha256=module.sha256(__file__))
        (evidence/'ejecucion.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        def table(frame):
            clean=lambda v:str(v).replace('|','/').replace('\n',' ')
            return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(map(clean,row))+' |' for row in frame.itertuples(index=False,name=None))
        lines=['# Modelo logístico WoE 5.0','',
            f"Recibidas de correlación: {len(data['variables_received'])}. Excluidas IV: {data['excluded_iv']}.",
            f"Eliminadas: {data['eliminated']}. Finalistas: {len(state.variables)}.",
            'Convención comprobada en código y JSON: ln(%Buenos suavizados / %Malos suavizados); target indicador_default, 1=malo/default. Signo esperado negativo; intercepto exento.',
            'IV > 0.50 se excluye por política de revisión; no demuestra por sí solo fuga de información.','',
            '## Variables finales','',*['- '+v for v in state.variables],'',
            '## Resumen estadístico original','', '```text',state.model.summary().as_text(),'```','',
            '## Métricas reales','',table(metrics[['Segmento','N','AUC','Gini','KS','Recall','Precision','F1_score','Brier','Accuracy','Tasa_default_observada','Probabilidad_promedio']].round(6)),'',
            table(env['df_deltas_modelo'].round(6)),'',
            'Los deltas firmados y absolutos usan tolerancias de seguimiento configuradas antes del ajuste; no son pruebas estadísticas. TRAIN es rendimiento aparente de desarrollo. El umbral de clasificación 0.50 no es un umbral operativo optimizado.',
            'P-values e intervalos son inferencia condicionada al modelo; la selección iterativa puede volverlos optimistas. No se utilizó TEST/OOT para elegir variables ni umbrales.','',
            '## Selección y SHAP','',
            'En cada ajuste se validan convergencia, rango y condición del diseño/Hessiano, separación y finitud de inferencia. Se elimina una variable: mayor magnitud de signo contrario; después mayor p-value; después menor porcentaje SHAP. Empates alfabéticos. Después de cada eliminación se ajusta y se vuelve a revisar la política desde signos.',
            'SHAP usa LinearExplainer con el vector de coeficientes/intercepto del mismo Logit, background TRAIN y muestra TRAIN reproducibles. Se verifica identidad analítica y aditividad en log-odds. Sus porcentajes no son probabilidades ni pruebas de significancia.','',
            '## Archivos','',
            '- modelos/modelo_logit.pkl: resultado statsmodels serializado. modelos/coeficientes.json: intercepto y coeficientes portables ordenados.',
            '- matrices/: X finales y targets por segmento, índice _row_id. predicciones/: target, probabilidad y clase al umbral configurado.',
            '- reportes/: summary_statsmodels.txt, coeficientes finales/todas las iteraciones, historial, filtro IV, eliminadas por criterio, SHAP, métricas y deltas.',
            '- graficos/: ROC, KS, matrices de confusión, calibración, SHAP final, coeficientes IC y evolución.',
            '- configuracion_modelo.json: parámetros, convención y referencias/hashes B.3 y 4.0.',
            '- evidencias/: pruebas, validación de entradas, muestreo SHAP, hashes previos, ejecución y conteo de ajustes.','',
            f"Pruebas: {tests_data}. Archivos previos verificados intactos: {len(protected)}. Ajustes: {len(fit_calls)}, todos TRAIN. Modelo recuperado reproduce las tres probabilidades con rtol=atol=1e-12.",'',
            '## Reproducción','',
            'Notebook: ../5.0.0.SCR_DES_MODELO_LOGISTICO.ipynb; módulo: ../5.0.0.modelo_logistico.py; pruebas: ../test_modelo_logistico.py.',
            '```powershell','.\\.venv\\Scripts\\python.exe 5.0.1.ejecutar_modelo_logistico.py','```','',
            'El ejecutor usa Python secuencial, captura display/plt.show y guarda el notebook ejecutado. No declara ejecución mediante kernel Jupyter. No ejecuta etapas previas ni recalcula cortes, WoE, IV, PSI o correlación. No se crean commits.']
        (ROOT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
        print(json.dumps({k:data[k] for k in ('variables_received','excluded_iv','eliminated','variables_final','metrics','deltas','tests')},indent=2,ensure_ascii=False))
    except Exception as exc:
        logs.append(traceback.format_exc())
        if getattr(exc,'state',None) is not None:
            exc.state.df_iteraciones_modelo.to_csv(evidence/'historial_detencion.csv',index=False)
            exc.state.df_coeficientes_iteraciones.to_csv(evidence/'coeficientes_detencion.csv',index=False)
        (evidence/'error_ejecucion.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        ipd.display,plt.show=original_display,original_show
        nbformat.write(notebook,NOTEBOOK)
        (evidence/'ejecucion.txt').write_text('\n'.join(logs),encoding='utf-8')
    (evidence/'notebook_sha256.txt').write_text(module.sha256(NOTEBOOK),encoding='utf-8')


if __name__=='__main__':main()
