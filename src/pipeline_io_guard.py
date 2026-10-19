"""Bloquea acceso al proyecto de origen durante la reproducción y sus subprocessos."""
from pathlib import Path
import atexit,json,os,sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'configs/pipeline.json').exists())
DENIED=[]
def guard(event,args):
    if event not in ('open','os.scandir','os.listdir','shutil.copyfile'):return
    for value in args[:2] if event=='shutil.copyfile' else args[:1]:
        if not isinstance(value,(str,bytes,os.PathLike)):continue
        text=os.fsdecode(value)
        if any(part.casefold()=='trabajofinal' for part in Path(text).parts):
            DENIED.append(dict(event=event,original_project_access_blocked=True))
            raise PermissionError('La reproducción solo puede usar el repositorio; acceso al proyecto original bloqueado')
sys.addaudithook(guard)
def finish():
    folder=ROOT/'outputs/access_audit';folder.mkdir(parents=True,exist_ok=True)
    (folder/f'process_{os.getpid()}.json').write_text(json.dumps(dict(process_id=os.getpid(),original_project_io_blocked=True,denied_access_attempts=DENIED)),encoding='utf-8')
atexit.register(finish)
