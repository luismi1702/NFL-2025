"""Cabecera de X (1500x500) de @CuartayDato con el estilo de docs/estilo-visual.md.
Sin el nombre de la cuenta (Luis, 02-oct-2026: "no quiero ver el nombre").

Dos versiones con datos reales de la temporada en curso:
  A  los 32 logos repartidos a todo lo ancho: EPA por jugada en ataque (x)
     contra EPA permitido en defensa (y, invertido: arriba = mejor defensa)
  B  una frase a la izquierda y los 32 como puntos de su color a la derecha
La esquina inferior izquierda queda despejada: ahi va la foto de perfil.

    python marca/cabecera_x.py        # desde la raiz del repo
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.offsetbox import AnnotationBbox      # noqa: E402
from matplotlib.patches import Rectangle              # noqa: E402
from pbp_loader import cargar_pbp                     # noqa: E402
from estilo import (BG, FG, GRID, GRIS1, GRIS2, DISP, TXT, color_equipo, logo,  # noqa: E402
                    texto_rico)

DIR = os.path.dirname(os.path.abspath(__file__))
df, season = cargar_pbp()
pr = df[df.play_type.isin(["pass", "run"]) & df.epa.notna()]
of = pr.groupby("posteam").epa.mean()
de = pr.groupby("defteam").epa.mean()
equipos = sorted(set(of.index) & set(de.index))


def campo(fig, hasta=11):
    """Lineas de yardas muy tenues."""
    for i in range(1, 12):
        fig.add_artist(Rectangle((i / 12, 0), 0.0012, 1, transform=fig.transFigure,
                                 color=GRID, alpha=0.5, lw=0))
    for i, yd in zip(range(2, hasta + 1, 2), [10, 20, 30, 40, 50]):
        for y in (0.08, 0.90):
            fig.text(i / 12, y, str(yd), family=DISP, weight="bold", size=17, color=GRID,
                     alpha=0.8, ha="center", va="center", rotation=0 if y < 0.5 else 180)


def nube(ax, logos):
    ax.set_facecolor("none")
    ax.axis("off")
    ax.axvline(of.mean(), color=GRID, lw=1)
    ax.axhline(de.mean(), color=GRID, lw=1)
    for t in equipos:
        if logos:
            im = logo(t, 0.034)
            if im is not None:
                ax.add_artist(AnnotationBbox(im, (of[t], de[t]), frameon=False, zorder=3))
                continue
        ax.scatter(of[t], de[t], s=150, color=color_equipo(t), edgecolors=BG, linewidths=1.5,
                   zorder=3)
    pad_x = (of.max() - of.min()) * 0.06
    pad_y = (de.max() - de.min()) * 0.10
    ax.set_xlim(of.min() - pad_x, of.max() + pad_x)
    ax.set_ylim(de.max() + pad_y, de.min() - pad_y)        # arriba = mejor defensa


# ── A: los 32 logos a todo lo ancho
fig = plt.figure(figsize=(15, 5), dpi=100, facecolor=BG)
campo(fig, hasta=10)
nube(fig.add_axes([0.24, 0.12, 0.72, 0.76]), logos=True)
fig.text(0.97, 0.05, f"Los 32 · EPA en ataque y en defensa · NFL {season}", family=TXT,
         size=9, color=GRIS2, ha="right", va="center")
fig.savefig(os.path.join(DIR, "cabecera_x_A.png"), dpi=100, facecolor=BG)
plt.close(fig)

# ── B: una frase y los 32 en puntos
fig = plt.figure(figsize=(15, 5), dpi=100, facecolor=BG)
campo(fig, hasta=6)
fig.text(0.215, 0.62, "La NFL,\ncontada con datos.", family=DISP, weight=800, size=58,
         color=FG, va="center", linespacing=0.95)
texto_rico(fig, 0.218, 0.33, [("Previas", GRIS1, "semibold"), ("·", GRIS2, "normal"),
                              ("análisis de cada partido", GRIS1, "semibold"),
                              ("·", GRIS2, "normal"), ("Manning Bot", "#ff8a1f", "bold")],
           size=15)
nube(fig.add_axes([0.66, 0.15, 0.31, 0.70]), logos=False)
fig.text(0.97, 0.06, f"Los 32 · EPA en ataque y en defensa · NFL {season}", family=TXT,
         size=9, color=GRIS2, ha="right", va="center")
fig.savefig(os.path.join(DIR, "cabecera_x_B.png"), dpi=100, facecolor=BG)
plt.close(fig)
print("Guardado: marca/cabecera_x_A.png y marca/cabecera_x_B.png")
