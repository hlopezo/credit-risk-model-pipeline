from pathlib import Path
import base64,contextlib,io,json,sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from pipeline_runtime import prepare_runtime

def execute_phase1():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt,nbformat,IPython.display as ipd
    path=REPO/'notebooks/1.0.0.SCR_DES_MOD_RL_SEGMT.ipynb';nb=nbformat.read(path,as_version=4)
    current=None;old_display,old_show=ipd.display,plt.show;env={'__name__':'__main__'};logs=[]
    def display(*items,**kwargs):
        for obj in items:
            data={'text/plain':repr(obj)}
            if hasattr(obj,'_repr_html_'):
                html=obj._repr_html_()
                if html:data['text/html']=html
            current.outputs.append(nbformat.v4.new_output('display_data',data=data))
    def show(*args,**kwargs):
        for number in plt.get_fignums():
            buffer=io.BytesIO();fig=plt.figure(number);fig.savefig(buffer,format='png',dpi=100)
            current.outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(buffer.getvalue()).decode(),'text/plain':repr(fig)}))
        plt.close('all')
    from sklearn import model_selection
    original_split=model_selection.train_test_split;split_calls=[]
    def observed_split(*args,**kwargs):
        development=args[0]
        if not development.periodo_observacion.le(202507).all():raise ValueError('Split fuera de la ventana original')
        if kwargs.get('test_size')!=.30 or kwargs.get('random_state')!=123:raise ValueError('Parámetros de split distintos al original')
        import pandas as pd
        pd.testing.assert_series_equal(kwargs['stratify'],development.indicador_default,check_exact=True)
        split_calls.append(dict(function='sklearn.model_selection.train_test_split',N_development=len(development),test_size=kwargs['test_size'],random_state=kwargs['random_state'],stratify='indicador_default',periods=sorted(map(int,development.periodo_observacion.unique()))))
        return original_split(*args,**kwargs)
    model_selection.train_test_split=observed_split
    ipd.display,plt.show=display,show
    env["display"]=display
    try:
        i=0
        for cell in nb.cells:
            if cell.cell_type!='code':continue
            current=cell;cell.outputs=[];i+=1;stream=io.StringIO();print('SEG',i,flush=True)
            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):exec(compile(cell.source,f'SEG_{i}','exec'),env)
            cell.execution_count=i
            if stream.getvalue():cell.outputs.insert(0,nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue()))
            logs.append(stream.getvalue())
    finally:
        model_selection.train_test_split=original_split
        ipd.display,plt.show=old_display,old_show
        output=REPO/'outputs/executed_notebooks';output.mkdir(parents=True,exist_ok=True);nbformat.write(nb,output/path.name)
        (output/'segmentacion_ejecucion.txt').write_text('\n'.join(logs),encoding='utf-8')
    assert len(split_calls)==1
    evidence=REPO/'outputs/executed_notebooks/phase1_split_calls.json'
    evidence.write_text(json.dumps(split_calls,indent=2),encoding='utf-8')
    return env['phase1_validation']
if __name__=='__main__':execute_phase1()
