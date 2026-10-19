"""Entrada compatible: delega B.3 al orquestador portable sin duplicar algoritmos."""
from pathlib import Path
import subprocess,sys
if __name__=='__main__':
    root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'configs/pipeline.json').exists())
    subprocess.run([sys.executable,str(root/'scripts/run_pipeline.py'),'--phases','3'],check=True)
