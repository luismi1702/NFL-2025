"""
vista_runner.py — Ejecuta un script del proyecto generando los PNGs en una
carpeta temporal (_preview/) en vez de en la raiz.

Usado por server.py para el boton "Ver analisis" del dashboard: el grafico
se ve exactamente igual que al generarlo, pero no se guarda nada en la
carpeta del proyecto (la vista previa se sobreescribe en cada ejecucion).

Uso: python vista_runner.py <script.py> [args...]
La carpeta destino se lee de la variable de entorno VISTA_DIR
(por defecto ./_preview).
"""
import os
import sys
import runpy

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

PREVIEW_DIR = os.environ.get("VISTA_DIR", os.path.join(os.getcwd(), "_preview"))
os.makedirs(PREVIEW_DIR, exist_ok=True)

_orig_savefig = Figure.savefig


def _savefig_preview(self, fname=None, *args, **kwargs):
    # Redirige rutas de archivo a _preview/ conservando el nombre.
    # plt.savefig tambien pasa por aqui (delega en Figure.savefig).
    if isinstance(fname, (str, os.PathLike)):
        base  = os.path.basename(str(fname))
        fname = os.path.join(PREVIEW_DIR, base)
        print(f"[modo vista] Vista previa temporal: {base} (no guardado en la carpeta)")
    return _orig_savefig(self, fname, *args, **kwargs)


Figure.savefig = _savefig_preview

if len(sys.argv) < 2:
    sys.exit("Uso: python vista_runner.py <script.py>")

script   = sys.argv[1]
sys.argv = sys.argv[1:]          # el script ve sus propios argv
runpy.run_path(script, run_name="__main__")
