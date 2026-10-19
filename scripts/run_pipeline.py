"""Ejecutor integral desde la fuente incluida; sin rutas del proyecto de origen."""
from pathlib import Path
import argparse,base64,contextlib,importlib.util,io,json,os,subprocess,sys,time,urllib.request
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from pipeline_runtime import prepare_runtime
import pipeline_io_guard
from segmentation_validation import require_generated_partition

def call(script,*args):
    subprocess.run([sys.executable,'-X','utf8',script,*args],check=True)

def execute_bivariate():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt,IPython.display as ipd,nbformat
    path=Path('3.0.0.SCR_DES_ANLS_BIVARIADO.ipynb');nb=nbformat.read(path,as_version=4)
    env={'__name__':'__main__'};current=None;logs=[];old_display,old_show=ipd.display,plt.show
    def display(*objects,**kwargs):
        for obj in objects:
            data={'text/plain':repr(obj)}
            if hasattr(obj,'_repr_html_'):
                html=obj._repr_html_()
                if html:data['text/html']=html
            current.outputs.append(nbformat.v4.new_output('display_data',data=data))
    def show(*args,**kwargs):
        for num in plt.get_fignums():
            buffer=io.BytesIO();fig=plt.figure(num);fig.savefig(buffer,format='png',dpi=90)
            current.outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(buffer.getvalue()).decode(),'text/plain':repr(fig)}))
        plt.close('all')
    ipd.display,plt.show=display,show
    try:
        count=0
        for cell in nb.cells:
            if cell.cell_type!='code':continue
            current=cell;cell.outputs=[];section=cell.metadata.get('biv_section','bootstrap');print('BIV',section,flush=True)
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):exec(compile(cell.source,'BIV_'+str(section),'exec'),env)
            count+=1;cell.execution_count=count
            if stream.getvalue():cell.outputs.insert(0,nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue()))
            logs.append(str(section)+' OK\n'+stream.getvalue())
        summary=env['df_resumen_bivariado'];data=dict(methodology_version='B.3',input_version=env['muestras_biv']['manifest']['version'],states=summary.Estado_bivariado.value_counts().to_dict(),variables=len(summary),all_candidates=len(env['df_candidatos_binning']),execution='Full B.3 recalculation from bundled raw CSV')
        e=Path('4.ANALIS_UV_BV_MV/BIVARIADO/evidencias');(e/'ejecucion.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    finally:
        ipd.display,plt.show=old_display,old_show
        output=REPO/'outputs/executed_notebooks';output.mkdir(exist_ok=True);nbformat.write(nb,output/path.name)
        Path('4.ANALIS_UV_BV_MV/BIVARIADO/evidencias/ejecucion_paquete.txt').write_text('\n'.join(logs),encoding='utf-8')

def server(cfg):
    uri=cfg['mlflow']['tracking_uri']
    try:
        urllib.request.urlopen(uri+'/health',timeout=2);return
    except Exception:pass
    folder=Path('MLFLOW_LOCAL/evidencias');folder.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'MLFLOW_DISABLE_AGENT_HINT':'1','MLFLOW_DISABLE_TELEMETRY':'true'}
    process=subprocess.Popen([sys.executable,'-m','mlflow','server','--host','127.0.0.1','--port',str(cfg['mlflow']['port']),'--backend-store-uri','sqlite:///MLFLOW_LOCAL/mlflow.db','--artifacts-destination','./MLFLOW_LOCAL/artifacts','--serve-artifacts','--workers','1'],stdout=(folder/'server_stdout.log').open('w'),stderr=(folder/'server_stderr.log').open('w'),env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    (folder/'server.pid').write_text(str(process.pid))
    for attempt in range(60):
        try:urllib.request.urlopen(uri+'/health',timeout=1);return
        except Exception:time.sleep(1)
    raise RuntimeError('MLflow no inició; revisar log local')

def validate():
    import numpy as np,pandas as pd
    reference=json.loads((REPO/'docs/reference/expected_results.json').read_text())
    biv=json.loads(Path('4.ANALIS_UV_BV_MV/BIVARIADO/binning_woe_train.json').read_text())
    corr=json.loads(Path('4.ANALIS_UV_BV_MV/MULTIVARIADO/seleccion_correlacion.json').read_text())
    model=json.loads(Path('5.MODELO_LOGISTICO/modelos/coeficientes.json').read_text())
    psi=pd.read_csv('4.ANALIS_UV_BV_MV/PSI_TEMPORAL/reportes/df_resumen_psi.csv')
    assert psi.Variable.tolist()==reference['psi_variables']
    assert biv['approved_variables']==reference['bivariate_approved'] and biv['rejected_variables']==reference['bivariate_rejected']
    for v,expected in reference['woe'].items():
        actual=biv['variables'][v]
        for key in ('edges','category_groups','state'):assert actual[key]==expected[key],(v,key)
        assert set(actual['woe_by_bin'])==set(expected['woe_by_bin'])
        np.testing.assert_allclose(list(actual['woe_by_bin'].values()),list(expected['woe_by_bin'].values()),rtol=1e-12,atol=1e-12)
    assert corr['retained_variables']==reference['correlation_variables'] and model['variables']==reference['final_model']['variables']
    np.testing.assert_allclose([model['intercept']]+list(model['coefficients'].values()),[reference['final_model']['intercept']]+list(reference['final_model']['coefficients'].values()),rtol=1e-10,atol=1e-12)
    metrics=pd.read_csv('5.MODELO_LOGISTICO/reportes/df_metricas_modelo.csv',float_precision='round_trip');expected=pd.DataFrame(reference['metrics'])
    cols=['AUC','Gini','KS','Recall','Precision','F1_score','Brier','Accuracy']
    differences=metrics[cols]-expected[cols];np.testing.assert_allclose(metrics[cols],expected[cols],rtol=1e-10,atol=1e-12)
    segmentation=require_generated_partition(REPO)
    report=dict(reproduced=True,from_unsegmented_source=True,phase1_partition_matches=True,partition_evidence=segmentation,psi_variables=len(psi),bivariate_approved=len(biv['approved_variables']),bivariate_rejected=len(biv['rejected_variables']),correlation_variables=len(corr['retained_variables']),model_variables=len(model['variables']),woe_cuts_match=True,coefficients_match=True,max_abs_metric_difference=float(np.max(np.abs(differences.to_numpy()))),metrics=json.loads(metrics.to_json(orient='records',double_precision=15)),metadata_note='Dataset version changes because regenerated evidence uses new environment/paths; numeric source, bins and model are compared independently.')
    (REPO/'outputs/reproduction_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phases',nargs='+',choices=['1','2','3','4','5','5.1','mlflow','tests','validate'],default=['1','2','3','4','5','5.1','mlflow','tests','validate']);args=parser.parse_args()
    os.environ['PYTHONPATH']=str(REPO/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
    runtime=prepare_runtime();os.chdir(runtime);sys.path.insert(0,str(runtime));cfg=json.loads((REPO/'configs/pipeline.json').read_text())
    for phase in args.phases:
        print('PHASE',phase,flush=True)
        if phase=='1':
            call(str(REPO/'scripts/ejecutar_segmentacion.py'))
        elif phase=='2':
            require_generated_partition(REPO);call('ejecutar_psi_notebook.py')
        elif phase=='3':
            spec=importlib.util.spec_from_file_location('biv_export','3.0.0.bivariado.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.export_v4_from_psi();execute_bivariate()
        elif phase=='4':call('4.0.1.ejecutar_multivariado.py')
        elif phase=='5':call('5.0.1.ejecutar_modelo_logistico.py')
        elif phase=='5.1':call('5.1.1.ejecutar_evaluacion_umbrales.py')
        elif phase=='mlflow':server(cfg);call('registrar_pipeline_mlflow.py')
        elif phase=='tests':
            import unittest
            suite=unittest.defaultTestLoader.discover(str(runtime),pattern='test_*.py');stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
            (REPO/'outputs/tests.txt').write_text(stream.getvalue(),encoding='utf-8')
            summary=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors));(REPO/'outputs/tests.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
            if not result.wasSuccessful():print(stream.getvalue());raise RuntimeError('Pruebas fallidas')
        else:validate()
    executed=REPO/'outputs/executed_notebooks';executed.mkdir(exist_ok=True)
    for nb in runtime.glob('*.ipynb'):
        if not nb.name.startswith(('1.','3.')):
            import shutil
            source = (runtime/'4.ANALIS_UV_BV_MV/PSI_TEMPORAL/evidencias/notebook_ejecutado.ipynb') if nb.name.startswith('2.') else nb
            shutil.copy2(source if source.exists() else nb,executed/nb.name)
if __name__=='__main__':main()
