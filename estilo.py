"""Estilo visual comun de @CuartayDato (docs/estilo-visual.md).

Excepcion a la regla de scripts independientes, aprobada por Luis el
02-oct-2026: con 56 scripts que generan PNG, un cambio de estilo tiene que
tocarse en un solo sitio. Aqui va SOLO diseño (fuentes, colores, piezas de
maquetacion); los datos siguen en pbp_loader.py.

Uso tipico:
    from estilo import *
    fig = lienzo()
    titular(fig, "Mandarle el blitz a San Francisco ha sido un regalo.")
    contexto(fig, [("Contra el blitz acierta en el", None), ("76%", "fuerte"), ...])
    tarjeta(fig, 0.20, 0.735)
    ...
    pie(fig, "Blitz = 1+ blitzer (FTN)", sello(SEASON))
    guardar(fig, salida("mi_pieza_2026.png", SEASON))
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyBboxPatch, Rectangle

# ── Fuentes (fonts/, Google Fonts, OFL)
_DIR = os.path.dirname(os.path.abspath(__file__))
for _f in os.listdir(os.path.join(_DIR, "fonts")):
    if _f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(_DIR, "fonts", _f))
TXT, DISP = "Barlow", "Barlow Condensed"
plt.rcParams["font.family"] = TXT          # ejes, ticks y leyendas incluidos

# ── Paleta (CLAUDE.md) + grises de jerarquia
BG, CARD, FG, GRID, ACCENT = "#0f1115", "#151924", "#EDEDED", "#2a2f3a", "#2d6cdf"
RYG = ["#d84a4a", "#ffd166", "#06d6a0"]
GRIS1, GRIS2 = "#9aa0ab", "#5d6370"        # texto secundario / terciario
NEUTRO = "#3a404d"                         # barras y puntos que no son protagonistas
MARCA = "@CuartayDato"

# ── Lienzo
ANCHO, ALTO = 5.4, 6.75                    # 4:5 = 1080x1350 a 200 dpi
L, R = 0.07, 0.93                          # margenes laterales


def lienzo(ancho=ANCHO, alto=ALTO):
    return plt.figure(figsize=(ancho, alto), facecolor=BG)


def guardar(fig, ruta, dpi=200):
    """Lienzo fijo: bbox_inches='tight' descoloca los logos (AnnotationBbox)."""
    fig.savefig(ruta, dpi=dpi, facecolor=BG)
    plt.close(fig)
    print("Guardado:", ruta)
    return ruta


# ── Color
def _lum(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contraste(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def aclarar(h, t):
    c = np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)])
    c = c + (255 - c) * t
    return "#" + "".join(f"{int(x):02x}" for x in c)


def _es_gris(h):
    c = [int(h[i:i + 2], 16) for i in (1, 3, 5)]
    return max(c) - min(c) < 40


def _saturacion(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return max(c) - min(c)


# Excepciones: el algoritmo elegiria un color que no es el de su identidad
FIJOS = {"BAL": aclarar("#241773", 0.35), "BUF": aclarar("#00338D", 0.25)}


def _color_de(fila):
    """Color de identidad legible sobre CARD (contraste >= 3, WCAG):
    1. el principal, tal cual o aclarado como mucho un 20%;
    2. si no, el color de la marca mas vivo que llegue sin ser gris;
    3. si ninguno, el principal aclarado lo que haga falta."""
    if fila.name in FIJOS:
        return FIJOS[fila.name]
    p = fila.team_color
    for t in np.arange(0, 0.21, 0.05):
        c = aclarar(p, t)
        if contraste(c, CARD) >= 3 and not _es_gris(c):
            return c
    cands = [(_saturacion(c), c) for c in (fila.team_color2, fila.team_color3, fila.team_color4)
             if isinstance(c, str) and c.startswith("#") and not _es_gris(c)
             and contraste(c, CARD) >= 3]
    if cands:
        return max(cands)[1]
    for t in np.arange(0.25, 0.95, 0.05):
        if contraste(aclarar(p, t), CARD) >= 3:
            return aclarar(p, t)
    return FG


_COL = None


def equipos():
    """teams.parquet indexado por siglas; si falta el cache, lo descarga."""
    ruta = os.path.join("pbp_cache", "teams.parquet")
    if not os.path.exists(ruta):
        from pbp_loader import cargar_equipos
        cargar_equipos()
    return pd.read_parquet(ruta).set_index("team_abbr")


def color_equipo(team):
    """Color de identidad de un equipo (siglas nflverse: LA, LV, WAS...)."""
    global _COL
    if _COL is None:
        eq = equipos()
        _COL = {t: _color_de(f) for t, f in eq.iterrows()}
    return _COL.get(team, FG)


def color_rango(rk, n=32):
    """Juicio SOLO en los extremos: top 5 verde, ultimos 5 rojo, resto gris."""
    return RYG[2] if rk <= 5 else RYG[0] if rk > n - 5 else GRIS1


# ── Numeros
def num(v, dec=2, signo=True):
    """Coma decimal y signo menos de verdad: −0,13 / +0,22."""
    s = f"{abs(v):.{dec}f}".replace(".", ",")
    if round(v, dec) == 0:
        return s
    if v < 0:
        return "−" + s
    return ("+" + s) if (signo and v > 0) else s


def pct(v, dec=0):
    """v en tanto por uno -> '46%'."""
    return f"{v * 100:.{dec}f}".replace(".", ",") + "%"


def ordinal(rk, n=32):
    return f"{rk}º de {n}"


# ── Logos (recorte por tinta, como load_logo de ranking_wrs.py)
def logo(team, zoom=0.02):
    ruta = os.path.join("logos", f"{team}.png")
    if not os.path.exists(ruta):
        return None
    img = plt.imread(ruta)
    if img.ndim == 3 and img.shape[2] == 4:
        ys, xs = np.where(img[:, :, 3] > 0.02)
        if len(ys):
            img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = img.shape[:2]
    z = zoom * 500.0 / max((h * w) ** 0.5, 1.0)
    if w * z > 900.0 * zoom:
        z = 900.0 * zoom / w
    return OffsetImage(img, zoom=z, resample=True)


def poner_logo(fig, team, x, y, zoom=0.02):
    im = logo(team, zoom)
    if im is not None:
        fig.add_artist(AnnotationBbox(im, (x, y), xycoords="figure fraction", frameon=False))


# ── Texto
def titular(fig, txt, y=0.94, x=L, size=23, ancho=None):
    """Nivel 1: la conclusion. Se parte solo por palabras al ancho del
    lienzo (o con \\n a mano). Devuelve la y por debajo del titular."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ancho = ancho or (R - x)
    lineas, actual = [], ""
    for parrafo in txt.split("\n"):
        actual = ""
        for palabra in parrafo.split(" "):
            prueba = (actual + " " + palabra).strip()
            t = fig.text(0, 0, prueba, family=DISP, weight="bold", size=size)
            w = t.get_window_extent(renderer=r).width / fig.bbox.width
            t.remove()
            if w > ancho and actual:
                lineas.append(actual)
                actual = palabra
            else:
                actual = prueba
        lineas.append(actual)
    t = fig.text(x, y, "\n".join(lineas), family=DISP, weight="bold", size=size, color=FG,
                 va="top", linespacing=1.0)
    alto = t.get_window_extent(renderer=r).height / fig.bbox.height
    return y - alto


def antetitulo(fig, txt, y=0.955, x=L):
    """Etiqueta pequeña sobre el titular (DATO DE LA SEMANA 3, PREVIA · SEM. 4)."""
    return fig.text(x, y, txt.upper(), family=TXT, weight="semibold", size=8, color=GRIS1,
                    va="top")


def etiqueta(fig, txt, x, y, ha="left"):
    """Nivel 2: que mide el numero, en versalitas."""
    return fig.text(x, y, txt.upper(), family=TXT, weight="medium", size=7.2, color=GRIS1,
                    va="center", ha=ha)


def nota(fig, txt, x, y, ha="left"):
    """Nivel 3: muestra, media, definicion."""
    return fig.text(x, y, txt, family=TXT, size=7.5, color=GRIS2, va="center", ha=ha)


def numero(fig, txt, x, y, size=46, color=FG, ha="left"):
    """Nivel 1: el dato estrella."""
    return fig.text(x, y, txt, family=DISP, weight=800, size=size, color=color, va="center",
                    ha=ha)


def texto_rico(fig, x, y, trozos, size=9.5, ancho=None, interlinea=1.45, family=TXT):
    """Escribe [(texto, color, peso), ...] seguidos, midiendo cada trozo y
    partiendo por palabras si pasa de `ancho` (fraccion de figura). Devuelve
    la y de la linea siguiente."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W = fig.bbox.width
    alto = size * interlinea / 72 * fig.dpi / fig.bbox.height
    x0 = x
    for k, (txt, color, peso) in enumerate(trozos):
        for i, palabra in enumerate(txt.split(" ")):
            pieza = palabra if (i == 0 and k == 0) else " " + palabra
            t = fig.text(x, y, pieza, family=family, size=size, color=color, weight=peso, va="top")
            w = t.get_window_extent(renderer=r).width / W
            if ancho and x + w > x0 + ancho and x > x0:
                t.remove()
                x, y = x0, y - alto
                t = fig.text(x, y, pieza.lstrip(), family=family, size=size, color=color,
                             weight=peso, va="top")
                w = t.get_window_extent(renderer=r).width / W
            x += w
    return y - alto


def contexto(fig, trozos, y=0.83, x=L, ancho=None):
    """Linea bajo el titular. trozos = [(texto, None|'fuerte'), ...]: lo
    'fuerte' va en FG negrita (las cifras), lo demas en GRIS1."""
    return texto_rico(fig, x, y, [(t, FG, "bold") if e == "fuerte" else (t, GRIS1, "normal")
                                  for t, e in trozos], size=9.5, ancho=ancho or (R - x))


# ── Bloques
def tarjeta(fig, y0, y1, x0=L - 0.02, x1=R + 0.02):
    """Fondo CARD con esquinas redondeadas, DEBAJO de cualquier eje."""
    fig.add_artist(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                  boxstyle="round,pad=0,rounding_size=0.015",
                                  transform=fig.transFigure, fc=CARD, ec="none", zorder=-1))


def linea_h(fig, y, x0=L, x1=R):
    fig.add_artist(Rectangle((x0, y), x1 - x0, 0.0015, transform=fig.transFigure, color=GRID,
                             lw=0))


def linea_v(fig, x, y0, y1):
    fig.add_artist(Rectangle((x, y0), 0.0015, y1 - y0, transform=fig.transFigure, color=GRID,
                             lw=0))


def pista32(fig, x, y, w, rk, n=32, color=None):
    """n casillas, la 1 a la izquierda; la del puesto, con el color de juicio."""
    cw = w / n
    for i in range(n):
        fig.add_artist(Rectangle((x + i * cw + cw * 0.14, y), cw * 0.72, 0.009,
                                 transform=fig.transFigure, color=GRID, lw=0))
    fig.add_artist(Rectangle((x + (rk - 1) * cw, y - 0.004), cw, 0.017,
                             transform=fig.transFigure, color=color or color_rango(rk, n), lw=0))


def pie(fig, definicion, sello_txt, y=0.045):
    """Definiciones y sello a la izquierda, marca de agua a la derecha."""
    if definicion:
        fig.text(L - 0.02, y + 0.02, definicion, family=TXT, size=6.5, color=GRIS2,
                 fontstyle="italic")
    fig.text(L - 0.02, y, sello_txt, family=TXT, size=6.5, color=GRIS2, fontstyle="italic")
    fig.text(R + 0.02, y, MARCA, family=TXT, size=8, color="#888888", alpha=0.8,
             fontstyle="italic", ha="right")


def estilo_ejes(ax, x=True, y=False):
    """Ejes limpios sobre la tarjeta: sin bordes, ticks en GRIS2, rejilla GRID."""
    ax.set_facecolor(CARD)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=GRIS2, labelsize=7, length=0)
    if not x:
        ax.set_xticks([])
    if not y:
        ax.set_yticks([])
    return ax


def _dist(a, b):
    ca = np.array([int(a[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    cb = np.array([int(b[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    return float(np.sqrt(((ca - cb) ** 2).sum()))


def colores_pareja(a, b, umbral=90):
    """Colores de dos equipos que van juntos (ARI-SF, KC-TB...). Si se
    parecen, el segundo pasa al color de su marca mas distinto del primero
    que siga siendo legible. Devuelve {a: color, b: color}."""
    ca, cb = color_equipo(a), color_equipo(b)
    if _dist(ca, cb) >= umbral:
        return {a: ca, b: cb}
    eq = equipos()
    cands = []
    if b in eq.index:
        f = eq.loc[b]
        for c in (f.team_color, f.team_color2, f.team_color3, f.team_color4):
            if isinstance(c, str) and c.startswith("#") and not _es_gris(c):
                for t in (0, 0.2, 0.35):
                    cc = aclarar(c, t)
                    if contraste(cc, CARD) >= 3:
                        cands.append(cc)
                        break
    mejor = max(cands, key=lambda c: _dist(c, ca), default=cb)
    return {a: ca, b: mejor if _dist(mejor, ca) > _dist(cb, ca) else cb}
