# match_preview_ranks_dark.py — ATAQUE, DEFENSA y EQUIPOS ESPECIALES (ranking 1=mejor)
# Modo oscuro, PNG listo para redes. Arriba SOLO logos (sin siglas).
# Fuente: nflverse play_by_play_2025 (lectura online).

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")   # consola cp1252: los caracteres > y >= la rompen
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pbp_loader import cargar_pbp, salida, season_cli, week_cli, orden_partido, sello
from estilo import (lienzo, guardar, antetitulo, titular, tarjeta, poner_logo, etiqueta,
                    linea_h, color_equipo, colores_pareja, color_rango, num, pie as pie_fn, L, R, TXT,
                    FG, GRIS1, GRIS2, DISP, cabecera, Y_BAJO_CABECERA)
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# === Config ===
SEASON       = season_cli()   # None = auto-detectar última temporada
EXP_PASS_YDS = 15   # jugada explosiva pase
EXP_RUN_YDS  = 10   # jugada explosiva carrera
FIGSIZE      = (8.7, 13.5)
DPI          = 170

# ---------------- Nombre del PNG ----------------
def nombre_preview(away, home, season, week, sufijo=""):
    """preview_... con el indice de kickoff delante (01 = TNF), como las fichas
    y los resumenes, para que la carpeta de la semana se lea en orden de partido."""
    orden = orden_partido(season, week, away, home) if week else None
    if orden:
        idx, vis, loc = orden
        return f"{idx:02d}_preview_{vis}_vs_{loc}{sufijo}_{season}.png"
    return f"preview_{away}_vs_{home}{sufijo}_{season}.png"

# ---------------- Helpers base ----------------
def success_rate(s: pd.Series) -> float:
    e = pd.to_numeric(s, errors="coerce").dropna()
    return float((e > 0).mean()*100) if len(e) else np.nan

def explosive_mask(df_sub: pd.DataFrame) -> pd.Series:
    y = pd.to_numeric(df_sub["yards_gained"], errors="coerce")
    return (df_sub["play_type"].eq("pass") & (y >= EXP_PASS_YDS)) | \
           (df_sub["play_type"].eq("run")  & (y >= EXP_RUN_YDS))

def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# --------- Cálculo ofensivo / defensivo ----------
def tercer_down(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Conversiones en 3er down por equipo (col = posteam o defteam), con las
    marcas de nflverse: un 3er down cuenta si acaba convertido o fallado."""
    t = df[(df["third_down_converted"] == 1) | (df["third_down_failed"] == 1)]
    g = t.groupby(col)["third_down_converted"].agg(hechos="sum", intentos="size")
    g["pct"] = g["hechos"] / g["intentos"] * 100
    return g


def compute_offense(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["play_type"].isin(["pass","run"])].copy()
    g = sub.groupby("posteam", dropna=True)
    out = pd.DataFrame(index=g.size().index)
    out["EPA/jugada"]        = g["epa"].mean()
    out["Éxito (%)"]         = g["epa"].apply(success_rate)
    out["EPA/pase"]          = sub.loc[sub["play_type"]=="pass"].groupby("posteam")["epa"].mean()
    out["EPA/carrera"]       = sub.loc[sub["play_type"]=="run"].groupby("posteam")["epa"].mean()
    sub["explosive"]         = explosive_mask(sub)
    out["Explosivas (%)"]    = sub.groupby("posteam")["explosive"].mean().mul(100)
    out["EPA 1º down"]       = sub.loc[sub["down"]==1].groupby("posteam")["epa"].mean()
    out["EPA downs tardíos"] = sub.loc[sub["down"]>=3].groupby("posteam")["epa"].mean()
    out["Distancia media 3º down"] = sub.loc[sub["down"]==3].groupby("posteam")["ydstogo"].mean()
    out["3er down (%)"]      = tercer_down(df, "posteam")["pct"]
    return out

def compute_defense(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["play_type"].isin(["pass","run"])].copy()
    g = sub.groupby("defteam", dropna=True)
    out = pd.DataFrame(index=g.size().index)
    out["EPA/jugada permitido"]   = g["epa"].mean()
    out["Éxito permitido (%)"]    = g["epa"].apply(success_rate)
    out["EPA/pase permitido"]     = sub.loc[sub["play_type"]=="pass"].groupby("defteam")["epa"].mean()
    out["EPA/carrera permitido"]  = sub.loc[sub["play_type"]=="run"].groupby("defteam")["epa"].mean()
    sub["explosive"]              = explosive_mask(sub)
    out["Explosivas permitidas (%)"] = sub.groupby("defteam")["explosive"].mean().mul(100)
    out["3er down permitido (%)"] = tercer_down(df, "defteam")["pct"]
    return out

# --------- Drives: RZ y %RedZone ----------
def compute_drive_level(df: pd.DataFrame):
    # Sin serie asignada (algún extra point suelto) no es una visita: con
    # dropna=False contaba como una serie más (19-oct-2026: SF salía 19/22
    # en la red zone y eran 19/21)
    dsub = df[df["posteam"].notna() & df["drive"].notna()].copy()
    for col in ["game_id","drive","posteam","defteam","yardline_100","touchdown","td_team"]:
        if col not in dsub.columns:
            dsub[col] = np.nan
    to_num(dsub, ["yardline_100","posteam_score_pre","posteam_score_post"])

    groups = dsub.groupby(["game_id","drive"], dropna=False, sort=False)
    recs = []
    for (gid, drv), g in groups:
        posteam = g["posteam"].iloc[0]
        defteam = g["defteam"].iloc[0]
        if pd.isna(posteam):
            continue
        start_y = g["yardline_100"].iloc[0] if "yardline_100" in g.columns else np.nan
        start_y = np.clip(start_y, 1, 99) if pd.notna(start_y) else np.nan
        min_y   = g["yardline_100"].min() if "yardline_100" in g.columns else np.nan
        entered_rz = (pd.notna(min_y) and (min_y <= 20))

        if "td_team" in g.columns:
            td_for = g["td_team"].astype(str).eq(str(posteam)).any()
        else:
            td_for = (g["touchdown"].fillna(False) & g["posteam"].astype(str).eq(str(posteam))).any()

        recs.append({
            "game_id": gid, "drive": drv,
            "posteam": posteam, "defteam": defteam,
            "start_yardline_100": start_y,
            "entered_rz": entered_rz,
            "td_for": td_for
        })
    return pd.DataFrame.from_records(recs)

def compute_redzone_metrics(drv_df: pd.DataFrame):
    rz = drv_df[drv_df["entered_rz"]].copy()

    # ATAQUE: %RedZone = TD / visitas a RZ
    off = rz.groupby("posteam").agg(rz_trips=("entered_rz","size"),
                                    td_drives=("td_for","sum"))
    off["%RedZone"] = (off["td_drives"] / off["rz_trips"]) * 100.0

    # DEFENSA: %RedZone permitido
    deff = rz.groupby("defteam").agg(rz_trips_allowed=("entered_rz","size"),
                                     td_drives_allowed=("td_for","sum"))
    deff["%RedZone permitido"] = (deff["td_drives_allowed"] / deff["rz_trips_allowed"]) * 100.0

    return off[["%RedZone"]], deff[["%RedZone permitido"]]

# --------- Equipos especiales ----------
def compute_special_teams(df: pd.DataFrame) -> pd.DataFrame:
    FG_COL = "field_goal_result" if "field_goal_result" in df.columns else ("fg_result" if "fg_result" in df.columns else None)
    parts = []

    # FG%
    if FG_COL is not None:
        fg = df[(df["play_type"]=="field_goal") & df["posteam"].notna()].copy()
        if not fg.empty:
            fg["made"] = fg[FG_COL].astype(str).str.lower().eq("made")
            parts.append(fg.groupby("posteam")["made"].mean().mul(100).rename("FG%"))

    # EPA/jugada ST
    st_types = {"kickoff","kickoff_return","punt","punt_return","field_goal","extra_point"}
    st_sub = df[df["play_type"].isin(st_types) & df["posteam"].notna()].copy()
    if not st_sub.empty:
        parts.append(st_sub.groupby("posteam")["epa"].mean().rename("EPA/jugada ST"))

    st = pd.concat(parts, axis=1) if parts else pd.DataFrame()
    for c in ["FG%","EPA/jugada ST"]:
        if c not in st.columns: st[c] = np.nan
    return st[["FG%","EPA/jugada ST"]]

# ---------------- Rankings ----------------
def rank_dataframe(df: pd.DataFrame, better_high_cols: list[str], better_low_cols: list[str]) -> pd.DataFrame:
    ranks = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in better_high_cols:
            r = df[col].rank(ascending=False, method="min")
        else:
            r = df[col].rank(ascending=True, method="min")
        ranks[col] = r.astype("Int64")
    return ranks

def rank_to_score(rank_value, max_rank):
    if pd.isna(rank_value) or pd.isna(max_rank) or max_rank < 1:
        return np.nan
    rank_value = float(rank_value); max_rank = float(max_rank)
    return 1.0 if max_rank == 1 else (max_rank - rank_value) / (max_rank - 1)

def safe_max(series):
    try:
        m = series.max()
        return float(m) if pd.notna(m) else np.nan
    except Exception:
        return np.nan

# ---------------- Dibujo ----------------
def load_logo(team: str, base_zoom=0.12):
    path = os.path.join("logos", f"{team}.png")
    if not os.path.exists(path):
        return None, None
    try:
        img = plt.imread(path)
        # Recorta margenes transparentes: algunos archivos traen mucho aire
        # (NYJ: tinta 3768x1186 en lienzo 4096x4096) y sin recorte salen enanos
        if img.ndim == 3 and img.shape[2] == 4:
            ys, xs = np.where(img[:, :, 3] > 0.02)
            if len(ys):
                img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = img.shape[:2]
        # Normaliza por el area de tinta real; los wordmarks apaisados
        # pueden ensancharse hasta 1.8x para compensar su poca altura
        zoom = base_zoom * 500.0 / max((h * w) ** 0.5, 1.0)
        if w * zoom > 900.0 * (base_zoom):
            zoom = 900.0 * (base_zoom) / w
        return img, zoom
    except Exception:
        return None, None

def record_equipo(team):
    """W-L de temporada regular desde el cache de schedules (opcional)."""
    try:
        sch = pd.read_parquet("pbp_cache/schedules.parquet")
        for c in ("result", "season"):
            sch[c] = pd.to_numeric(sch[c], errors="coerce")
        reg = sch[(sch["season"] == SEASON) & (sch["game_type"] == "REG") &
                  sch["result"].notna()]
        h = reg[reg["home_team"] == team]
        a = reg[reg["away_team"] == team]
        w = int((h["result"] > 0).sum() + (a["result"] < 0).sum())
        l = int((h["result"] < 0).sum() + (a["result"] > 0).sum())
        e = int((h["result"] == 0).sum() + (a["result"] == 0).sum())
        return f"{w}-{l}" + (f"-{e}" if e else "")
    except Exception:
        return ""


# Cruces: cada metrica de ataque con su gemela de defensa.
# (columna ataque, columna defensa, etiqueta, formato, frase ataque, frase defensa)
CRUCES = [
    ("EPA/jugada", "EPA/jugada permitido", "EPA por jugada", "num",
     "en EPA por jugada", "en EPA permitido"),
    ("Éxito (%)", "Éxito permitido (%)", "Jugadas con éxito", "pct",
     "en jugadas con éxito", "en éxito permitido"),
    ("EPA/pase", "EPA/pase permitido", "EPA por pase", "num",
     "en EPA por pase", "contra el pase"),
    ("EPA/carrera", "EPA/carrera permitido", "EPA por carrera", "num",
     "en EPA por carrera", "contra la carrera"),
    ("Explosivas (%)", "Explosivas permitidas (%)", "Jugadas explosivas", "pct",
     "en jugadas explosivas", "en explosivas permitidas"),
    ("3er down (%)", "3er down permitido (%)", "Conversión en 3er down", "d3",
     "en conversión de 3er down", "defendiendo el 3er down"),
    ("%RedZone", "%RedZone permitido", "TD en red zone", "rz",
     "en TD en la red zone", "defendiendo su red zone"),
]
# Para el titular se prefieren los cruces concretos a los generales
PREFERENCIA = ["EPA/pase", "EPA/carrera", "3er down (%)", "Explosivas (%)", "%RedZone", "EPA/jugada", "Éxito (%)"]
EXTREMO = 8          # titular solo si los dos rangos estan en el top/bottom 8
MIN_RZ = 6           # visitas minimas a la red zone para usarla en el titular


def nombre_equipo(team):
    """Ciudad para el titular; los equipos que comparten ciudad, por su apodo."""
    try:
        eq = pd.read_parquet("pbp_cache/teams.parquet").set_index("team_abbr")
        if team in ("LA", "LAC", "NYG", "NYJ"):
            return "los " + eq.loc[team, "team_nick"]
        return eq.loc[team, "team_name"].replace(" " + eq.loc[team, "team_nick"], "")
    except Exception:
        return team


def mayor_desajuste(team_a, team_b, off_r, deff_r, rz_n):
    """El cruce mas desigual del partido, o None si ninguno es extremo.
    Devuelve (atacante, defensor, cruce, rk_ataque, rk_defensa)."""
    cands = []
    for atk, dfn in ((team_a, team_b), (team_b, team_a)):
        for c in CRUCES:
            co, cd = c[0], c[1]
            if co not in off_r.columns or cd not in deff_r.columns:
                continue
            if atk not in off_r.index or dfn not in deff_r.index:
                continue
            ro, rd = off_r.loc[atk, co], deff_r.loc[dfn, cd]
            if pd.isna(ro) or pd.isna(rd):
                continue
            ro, rd = int(ro), int(rd)
            if co == "%RedZone" and (rz_n.get(("of", atk), 0) < MIN_RZ
                                     or rz_n.get(("df", dfn), 0) < MIN_RZ):
                continue
            ventaja = ro <= EXTREMO and rd > 32 - EXTREMO
            ahogo = ro > 32 - EXTREMO and rd <= EXTREMO
            if ventaja or ahogo:
                cands.append((abs(ro - rd), -PREFERENCIA.index(co), atk, dfn, c, ro, rd))
    if not cands:
        return None
    _, _, atk, dfn, c, ro, rd = max(cands, key=lambda x: (x[0], x[1]))
    return atk, dfn, c, ro, rd


def frase_titular(d):
    """Plantilla del titular: solo rangos, que son hechos. El fuerte va primero."""
    atk, dfn, c, ro, rd = d
    A, B = nombre_equipo(atk), nombre_equipo(dfn)
    def ser(nombre):   # "los Rams son", "Denver es"
        return "son" if nombre.startswith("los ") else "es"
    if ro < rd:     # el ataque es el fuerte
        s = f"{A} {ser(A)} {ro}º {c[4]}; {B}, {rd}º {c[5]}."
    else:
        s = f"{B} {ser(B)} {rd}º {c[5]}; {A}, {ro}º {c[4]}."
    return s[0].upper() + s[1:]   # solo la primera: "...; los Giants, 25º"


def fmt_val(v, is_pct=False, nd=2):
    """Para el TXT del redactor (mismo formato de siempre)."""
    if pd.isna(v): return "–"
    return f"{v:.1f}%" if is_pct else f"{v:.{nd}f}"


def draw_png(team_a, team_b, off, deff, st, off_r, deff_r, st_r, out_path, pie="",
             rz_cnt=None, semana=None, d3_cnt=None):
    """Previa de un partido (team_a visita a team_b) como tarjeta de cruce
    (11-oct-2026, idea de SumerSports): cabecera con los dos logos sobre el
    color de cada equipo, titular con el mayor desajuste y los dos cruces con
    el ataque a la izquierda, la defensa a la derecha y la metrica en el centro."""
    rz_cnt = rz_cnt or {}
    cnt = {"rz": rz_cnt, "d3": d3_cnt or {}}
    rz_n = {k: v[1] for k, v in rz_cnt.items()}
    XM = 0.5
    fig = lienzo()
    recs = {t: (record_equipo(t), rol) for t, rol in ((team_a, "visitante"), (team_b, "local"))}
    COLP = cabecera(fig, (team_a, team_b),
                    {t: f"{rec} · {rol}" if rec else rol for t, (rec, rol) in recs.items()})

    antetitulo(fig, f"Previa · semana {semana}" if semana else "Previa", y=Y_BAJO_CABECERA)
    d = mayor_desajuste(team_a, team_b, off_r, deff_r, rz_n)
    y_top = (titular(fig, frase_titular(d), y=0.82, size=18.5) - 0.02) if d else 0.815
    destacado = (d[0], d[2][0]) if d else None

    # Dos cruces: cuando ataca el visitante y cuando ataca el local
    # Sin titular sobra sitio: se reparte entre las filas para que la tarjeta
    # llegue igual de abajo que en las previas que si lo llevan
    alto_fila = 0.0325 + max(0.0, y_top - 0.73) / 16
    y = y_top - 0.03
    for atk, dfn in ((team_a, team_b), (team_b, team_a)):
        poner_logo(fig, atk, L + 0.022, y, zoom=0.017)
        fig.text(L + 0.05, y, f"ATAQUE {atk}", family=TXT, weight="bold", size=7.5,
                 color=COLP[atk], va="center")
        poner_logo(fig, dfn, R - 0.022, y, zoom=0.017)
        fig.text(R - 0.05, y, f"DEFENSA {dfn}", family=TXT, weight="bold", size=7.5,
                 color=COLP[dfn], va="center", ha="right")
        fig.text(XM, y, "contra", family=TXT, size=7, color=GRIS2, va="center", ha="center")
        linea_h(fig, y - 0.021)
        y -= 0.043
        for co, cd, lab, formato, _, _ in CRUCES:
            if destacado == (atk, co):
                fig.add_artist(FancyBboxPatch((L, y - alto_fila / 2), R - L, alto_fila,
                                              boxstyle="round,pad=0,rounding_size=0.006",
                                              transform=fig.transFigure, fc="#1e2433",
                                              ec="none", zorder=-0.5))
            fig.text(XM, y, lab, family=TXT, size=8, color=GRIS1, va="center", ha="center")
            # sg: +1 dibuja hacia la derecha (ataque), -1 hacia la izquierda (defensa)
            for lado, team, tabla, rangos, c, ha, sg in ((("of", atk, off, off_r, co, "left", 1)),
                                                        (("df", dfn, deff, deff_r, cd, "right", -1))):
                x = L + 0.01 if ha == "left" else R - 0.01
                v = tabla.loc[team, c] if (team in tabla.index and c in tabla.columns) else np.nan
                rk = rangos.loc[team, c] if (team in rangos.index and c in rangos.columns) else np.nan
                txt = "–" if pd.isna(v) else num(v) if formato == "num" else f"{v:.0f}%"
                fig.text(x, y, txt, family=DISP, weight="bold", size=13, color=FG, va="center",
                         ha=ha)
                hechos, visitas = cnt.get(formato, {}).get((lado, team), (None, None))
                if visitas and pd.notna(v):
                    fig.text(x + sg * 0.075, y - 0.001, f"{hechos}/{visitas}", family=TXT,
                             size=6.5, color=GRIS2, va="center", ha=ha)
                if pd.notna(rk):
                    fig.text(XM - sg * 0.185, y, f"{int(rk)}º", family=TXT, weight="bold",
                             size=8.5, color=color_rango(int(rk)), va="center",
                             ha="right" if ha == "left" else "left")
            y -= alto_fila
        y -= 0.022

    # Especiales en una linea
    def esp(t):
        fg = st.loc[t, "FG%"] if t in st.index else np.nan
        rk = st_r.loc[t, "FG%"] if t in st_r.index else np.nan
        return f"{t} {fg:.0f}% ({int(rk)}º)" if pd.notna(fg) and pd.notna(rk) else f"{t} –"
    linea_h(fig, y + 0.014)
    etiqueta(fig, "Field goals", XM, y - 0.008, ha="center")
    fig.text(L + 0.01, y - 0.008, esp(team_a), family=TXT, size=8, color=GRIS1, va="center")
    fig.text(R - 0.01, y - 0.008, esp(team_b), family=TXT, size=8, color=GRIS1, va="center",
             ha="right")
    tarjeta(fig, y - 0.035, y_top)

    pie_fn(fig, "Rango 1 = el mejor de 32 (en defensa, el que menos concede) · Éxito = EPA > 0 · "
                "Explosiva = pase de 15+ o carrera de 10+", pie)
    guardar(fig, out_path)
    return fig


# ---------------- Main ----------------
# ---------------- Radar: las claves del partido (2º PNG, 11-oct-2026) ----------------
# Nace de un tuit con los screens de San Francisco contra Seattle que se nos escapo
# en un duelo del sabado: el dato estaba en FTN y nadie pregunto por el. El radar
# repasa ~45 situaciones por partido y se queda con los cruces mas desiguales.
RADAR_MIN_N = 10    # jugadas minimas de cada lado para que el cruce cuente
N_CLAVES    = 5     # las que caben en el PNG
BASE_OF  = {"db": "sus pases", "run": "sus carreras", "all": "sus jugadas"}
BASE_DEF = {"db": "los pases que defiende", "run": "las carreras que defiende",
            "all": "las jugadas que defiende"}
POS_TXT  = {"WR": "wide receivers", "TE": "tight ends", "RB": "running backs"}
PERSONAL = {"10": "1 RB, 0 TE", "11": "1 RB, 1 TE", "12": "1 RB, 2 TE", "13": "1 RB, 3 TE",
            "20": "2 RB, 0 TE", "21": "2 RB, 1 TE", "22": "2 RB, 2 TE"}


def _com(x, dec=1):
    return f"{x:.{dec}f}".replace(".", ",")


def _epa(x):
    return f"{x:+.2f}".replace(".", ",")


def construir_radar(d, season, semana):
    """Todas las tablas de cruce (ataque y defensa, con su rango) a partir de las
    jugadas `d`, que ya vienen cortadas en la semana anterior. Cada fuente
    auxiliar que falte (FTN, roster, PFR, datos_web) se avisa y se salta."""
    from pbp_loader import (cargar_ftn, cargar_rosters, cargar_pfr, cargar_web,
                            DatosNoDisponibles)
    fuentes = set()
    p = d[d.play_type.isin(["pass", "run"]) & d.epa.notna() & d.posteam.notna()
          & (d.qb_kneel != 1) & (d.qb_spike != 1)].copy()
    p["db"] = p.qb_dropback == 1
    p["run"] = (p.play_type == "run") & (p.qb_scramble != 1)
    p["all"] = True
    db, run, pase = p.db, p.run, p.play_type == "pass"
    air = pd.to_numeric(p.air_yards, errors="coerce")

    # nombre: (mascara, familia, base del % de uso, quien elige la situacion: of / df / None)
    sit = {
        "Pase en 1er down": (db & (p.down == 1), "1er down", "all", None),
        "Carrera en 1er down": (run & (p.down == 1), "1er down", "all", None),
        "3er down corto (1-3 yardas)": ((p.down == 3) & (p.ydstogo <= 3), "3er down", "all", None),
        "3er down largo (7+ yardas)": ((p.down == 3) & (p.ydstogo >= 7), "3er down", "all", None),
        "Zona roja": (p.yardline_100 <= 20, "zona roja", "all", None),
        "Dentro de la yarda 10": (p.yardline_100 <= 10, "zona roja", "all", None),
        "Pase desde shotgun": (db & (p.shotgun == 1), "formacion pase", "db", "of"),
        "Pase desde bajo centro": (db & (p.shotgun == 0), "formacion pase", "db", "of"),
        "Carrera desde shotgun": (run & (p.shotgun == 1), "formacion carrera", "run", "of"),
        "Carrera desde bajo centro": (run & (p.shotgun == 0), "formacion carrera", "run", "of"),
        "Pase profundo (20+ por aire)": (pase & (air >= 20), "profundidad", "db", "of"),
        "Pase corto (menos de 10 por aire)": (pase & (air < 10), "profundidad", "db", "of"),
        "Pase al centro del campo": (pase & (p.pass_location == "middle"), "zona pase", "db", "of"),
        "Pase a los lados": (pase & p.pass_location.isin(["left", "right"]), "zona pase", "db", "of"),
        "Carrera por dentro": (run & ((p.run_location == "middle") | (p.run_gap == "guard")),
                               "zona carrera", "run", "of"),
        "Carrera por fuera": (run & (p.run_gap == "end"), "zona carrera", "run", "of"),
        "Sin huddle": (p.no_huddle == 1, "ritmo", "all", "of"),
        "Últimos 2 minutos de cada parte": (p.half_seconds_remaining <= 120, "reloj", "all", None),
    }
    try:
        ftn = cargar_ftn(season)
        ftn = ftn[0] if isinstance(ftn, tuple) else ftn
        sems = sorted(d.week.dropna().unique())
        falta = (d.groupby("week").game_id.nunique()
                 - ftn.groupby("week").nflverse_game_id.nunique().reindex(sems).fillna(0))
        if (falta > 0).any():
            print("  AVISO radar: FTN incompleto, partidos que faltan por semana:",
                  falta[falta > 0].to_dict())
        f = ftn[ftn.qb_location.astype(str) != "0"]
        idx = p.index
        p = p.merge(f[["nflverse_game_id", "nflverse_play_id", "is_play_action", "is_motion",
                       "is_screen_pass", "is_rpo", "is_qb_out_of_pocket", "n_offense_backfield",
                       "n_blitzers", "n_pass_rushers", "n_defense_box"]],
                    left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"],
                    how="left")
        p.index = idx
        hay = p.nflverse_play_id.notna()
        nb = pd.to_numeric(p.n_blitzers, errors="coerce")
        nr = pd.to_numeric(p.n_pass_rushers, errors="coerce")
        box = pd.to_numeric(p.n_defense_box, errors="coerce")
        sit.update({
            "Screens": (hay & pase & (p.is_screen_pass == True), "screens", "db", "of"),            # noqa: E712
            "Play action": (hay & db & (p.is_play_action == True), "play action", "db", "of"),      # noqa: E712
            "Pase sin play action": (hay & db & (p.is_play_action == False), "play action", "db", "of"),  # noqa: E712
            "Pase con motion": (hay & db & (p.is_motion == True), "motion", "db", "of"),            # noqa: E712
            "Carrera con motion": (hay & run & (p.is_motion == True), "motion", "run", "of"),       # noqa: E712
            "RPO": (hay & (p.is_rpo == True), "rpo", "all", "of"),                                  # noqa: E712
            "QB fuera del pocket": (hay & db & (p.is_qb_out_of_pocket == True), "pocket", "db", None),  # noqa: E712
            "Backfield vacío": (hay & db & (pd.to_numeric(p.n_offense_backfield, errors="coerce") == 0),
                                "vacio", "db", "of"),
            "Pase contra blitz": (hay & db & (nb >= 1), "blitz", "db", "df"),
            "Pase sin blitz": (hay & db & (nb == 0), "blitz", "db", "df"),
            "Pase contra 3 rushers": (hay & db & (nr <= 3), "rushers", "db", "df"),
            "Carrera contra caja de 8+": (hay & run & (box >= 8), "caja", "run", "df"),
            "Carrera contra caja de 6 o menos": (hay & run & (box <= 6), "caja", "run", "df"),
        })
        fuentes.add("FTN")
    except DatosNoDisponibles as e:
        print("  Radar sin FTN: sale sin screens, play action, motion ni blitz.", e)

    def tabla_epa(q, col, n_base=None, unidad="jugadas"):
        t = q.groupby(col).agg(n=("epa", "size"), val=("epa", "mean"), yds=("yards_gained", "mean"))
        t["rk"] = t.val.rank(ascending=(col == "defteam"), method="min")
        t["txt"] = [f"{_epa(r.val)} EPA · {_com(r.yds)} yd · {int(r.n)} {unidad}" for r in t.itertuples()]
        if n_base is not None:
            t["uso"] = t.n / n_base
            t["rk_uso"] = t.uso.rank(ascending=False, method="min")
        return t

    def tabla_k(k, n, mejor_alta, unidad):
        """Claves que no son EPA: una tasa (hechos entre oportunidades)."""
        t = pd.DataFrame({"k": k, "n": n}).dropna()
        t["val"] = t.k / t.n
        t["yds"] = np.nan
        t["rk"] = t.val.rank(ascending=not mejor_alta, method="min")
        t["txt"] = [f"{_com(r.val * 100)}% · {int(r.k)} en {int(r.n)} {unidad}" for r in t.itertuples()]
        return t

    tab = {}
    for k, (m, fam, base, elige) in sit.items():
        q = p[m.fillna(False).astype(bool)]
        tab[k] = dict(of=tabla_epa(q, "posteam", p[p[base]].groupby("posteam").size() if elige == "of" else None),
                      df=tabla_epa(q, "defteam", p[p[base]].groupby("defteam").size() if elige == "df" else None),
                      fam=fam, base=base, elige=elige)

    # Series: three-and-outs, perdidas y series con puntos
    ser = d[d.posteam.notna() & d.fixed_drive.notna()].groupby(["game_id", "fixed_drive"]).agg(
        posteam=("posteam", "first"), defteam=("defteam", "first"),
        res=("fixed_drive_result", "first"), fd=("drive_first_downs", "first")).reset_index()
    ser["tao"] = (ser.res == "Punt") & (ser.fd == 0)
    ser["perdida"] = ser.res.isin(["Turnover", "Opp touchdown"])
    ser["puntos"] = ser.res.isin(["Touchdown", "Field goal"])
    dbs = p[p.db].assign(sack=lambda x: x.sack == 1)
    # (nombre, tabla, columna, unidad, familia, ¿en ataque mas es mejor?)
    for nombre, q, cuenta, unidad, fam, of_alta in (
            ("Three-and-outs", ser, "tao", "series", "series", False),
            ("Pérdidas de balón por serie", ser, "perdida", "series", "perdidas", False),
            ("Sacks por dropback", dbs, "sack", "dropbacks", "sacks", False),
            ("Series que acaban en puntos", ser, "puntos", "series", "puntos", True)):
        g_of, g_df = q.groupby("posteam")[cuenta], q.groupby("defteam")[cuenta]
        tab[nombre] = dict(of=tabla_k(g_of.sum(), g_of.size(), of_alta, unidad),
                           df=tabla_k(g_df.sum(), g_df.size(), not of_alta, unidad),
                           fam=fam, base="all", elige=None)

    # Receptores por posicion (PBP + roster) y el receptor mas buscado de cada equipo
    lider, def_pos = None, {}
    try:
        ros = cargar_rosters(season)
        ros = ros[0] if isinstance(ros, tuple) else ros
        pos = (ros.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id").position
               .replace({"FB": "RB"}))
        p["pos_rec"] = p.receiver_player_id.map(pos)
        tg = p[pase & p.receiver_player_id.notna()]
        n_tg = {"posteam": tg.groupby("posteam").size(), "defteam": tg.groupby("defteam").size()}
        for ps, nombre in POS_TXT.items():
            q = tg[tg.pos_rec == ps]
            t = {col: tabla_epa(q, col, n_tg[col], "targets") for col in ("posteam", "defteam")}
            tab[f"Pases a los {nombre}"] = dict(of=t["posteam"], df=t["defteam"], fam=f"pos {ps}",
                                                base="db", elige="of")
            def_pos[ps] = t["defteam"]
        j = tg.groupby(["posteam", "receiver_player_id"]).agg(
            nombre=("receiver_player_name", "first"), pos=("pos_rec", "first"), n=("epa", "size"),
            val=("epa", "mean"), yds=("yards_gained", "mean")).reset_index()
        j = j.sort_values("n", ascending=False).drop_duplicates("posteam").set_index("posteam")
        j["cuota"] = j.n / n_tg["posteam"]
        j["rk"] = j.val.rank(ascending=False, method="min")
        j["txt"] = [f"{r.cuota * 100:.0f}% de los targets · {_com(r.yds)} yd" for r in j.itertuples()]
        lider = j[j.pos.isin(POS_TXT)]
    except DatosNoDisponibles as e:
        print("  Radar sin roster: sale sin receptores por posicion.", e)

    # Presion y placajes fallados (PFR semanal, la misma fuente para los dos lados)
    try:
        pf, _ = cargar_pfr("def", season, semanal=True)
        pf = pf[(pf.week < semana) & (pf.game_type == "REG")]
        tab["Presión por dropback"] = dict(
            of=tabla_k(pf.groupby("opponent").def_pressures.sum(), dbs.groupby("posteam").size(), False, "dropbacks"),
            df=tabla_k(pf.groupby("team").def_pressures.sum(), dbs.groupby("defteam").size(), True, "dropbacks"),
            fam="presion", base="all", elige=None)
        tab["Placajes fallados"] = dict(
            of=tabla_k(pf.groupby("opponent").def_missed_tackles.sum(), p.groupby("posteam").size(), True, "jugadas"),
            df=tabla_k(pf.groupby("team").def_missed_tackles.sum(), p.groupby("defteam").size(), False, "jugadas"),
            fam="placajes", base="all", elige=None)
        fuentes.add("PFR")
    except DatosNoDisponibles as e:
        print("  Radar sin PFR semanal: sale sin presion ni placajes fallados.", e)

    # Personal (SumerSports): el ataque con cada agrupacion contra la defensa ante esa misma
    try:
        pa, _ = cargar_web("sumersports_personal_ataque", season, semana)
        pdf_, _ = cargar_web("sumersports_personal_defensa", season, semana)
        for per, desc in PERSONAL.items():
            t = {}
            for lado, tb in (("of", pa), ("df", pdf_)):
                x = tb[tb.personal.astype(str) == per].set_index("team")
                x = x[x.jugadas >= RADAR_MIN_N + 5]
                x = x.assign(n=x.jugadas, val=x.epa_jugada, yds=np.nan, uso=x.uso_pct / 100)
                x["rk"] = x.val.rank(ascending=(lado == "df"), method="min")
                x["rk_uso"] = x.uso.rank(ascending=False, method="min")
                x["txt"] = [f"{_epa(r.val)} EPA · {int(r.n)} jugadas" for r in x.itertuples()]
                t[lado] = x
            # con menos de 16 equipos el rango no dice nada
            if len(t["of"]) >= 16 and len(t["df"]) >= 16:
                tab[f"Personal {per} ({desc})"] = dict(of=t["of"], df=t["df"], fam="personal",
                                                       base="all", elige="of")
        fuentes.add("SumerSports")
    except DatosNoDisponibles as e:
        print("  Radar sin datos_web de SumerSports: sale sin personal.", e)

    # Trincheras en carrera y coberturas (Sharp): tablas de equipo, sin recuento de jugadas
    cob = cpos = sem_cob = None
    try:
        ol, _ = cargar_web("sharp_linea_ofensiva", season, semana)
        dl, _ = cargar_web("sharp_linea_defensiva", season, semana)
        ol, dl = ol.set_index("team"), dl.set_index("team")

        def tabla_sharp(v, mejor_alta, fmt):
            t = pd.DataFrame({"val": v, "n": 999, "yds": np.nan})
            t["rk"] = t.val.rank(ascending=not mejor_alta, method="min")
            t["txt"] = [fmt(x) for x in t.val]
            return t
        tab["Yardas antes del contacto (RB)"] = dict(
            of=tabla_sharp(ol["Yards Before Contact Per RB Rush"], True, lambda v: f"{_com(v)} yd por carrera"),
            df=tabla_sharp(dl["yardas_antes_contacto_rb"], False, lambda v: f"{_com(v)} yd por carrera"),
            fam="trinchera carrera", base="all", elige=None)
        tab["Carreras frenadas en la línea"] = dict(
            of=tabla_sharp(ol["Rush Stuff Rate"], False, lambda v: f"{_com(v)}% de sus carreras"),
            df=tabla_sharp(dl["stuff_pct"], True, lambda v: f"{_com(v)}% de las que defiende"),
            fam="trinchera carrera", base="all", elige=None)
        cob, sem_cob = cargar_web("sharp_coberturas", season, semana)
        cob = cob.set_index("team")
        cob["rk_man"] = cob.man_pct.rank(ascending=False, method="min")
        cpos, _ = cargar_web("sharp_cobertura_posicion", season, semana)
        cpos = cpos.set_index("team")
        cpos["rk_slot"] = cpos.ypt_slot.rank(method="min")
        fuentes.add("Sharp")
    except DatosNoDisponibles as e:
        print("  Radar sin datos_web de Sharp: sale sin trincheras ni coberturas.", e)

    return dict(tab=tab, lider=lider, def_pos=def_pos, cob=cob, cpos=cpos, sem_cob=sem_cob,
                fuentes=fuentes)


def claves_partido(radar, a, h):
    """Los cruces mas desiguales del partido, en los dos sentidos: como mucho uno
    por familia y sentido (zona roja y dentro de la 10 son la misma historia).
    Una de las claves es siempre el receptor mas buscado con el cruce mas desigual."""
    cand = []
    for atk, dfn in ((a, h), (h, a)):
        for k, t in radar["tab"].items():
            if atk not in t["of"].index or dfn not in t["df"].index:
                continue
            x, y = t["of"].loc[atk], t["df"].loc[dfn]
            if x.n < RADAR_MIN_N or y.n < RADAR_MIN_N:
                continue
            cand.append(dict(sit=k, atk=atk, dfn=dfn, x=x, y=y, dif=int(y.rk - x.rk),
                             fam=t["fam"], base=t["base"], elige=t["elige"]))
        lider = radar["lider"]
        if lider is not None and atk in lider.index:
            x = lider.loc[atk]
            if dfn in radar["def_pos"][x.pos].index and x.n >= RADAR_MIN_N:
                y = radar["def_pos"][x.pos].loc[dfn]
                cand.append(dict(sit=f"{x.nombre}, el más buscado de {atk}", atk=atk, dfn=dfn, x=x, y=y,
                                 dif=int(y.rk - x.rk), fam="jugador", base="all", elige=None,
                                 rol_of=f"{x.pos} · {int(x.n)} TARGETS",
                                 rol_df=f"DEFENSA {dfn} A LOS {x.pos}"))
    cand.sort(key=lambda c: (-abs(c["dif"]), -(c["x"].n + c["y"].n)))
    out, vistas = [], set()
    for c in cand:
        if (c["atk"], c["fam"]) in vistas:
            continue
        vistas.add((c["atk"], c["fam"]))
        out.append(c)
    if not any(c["fam"] == "jugador" for c in out[:N_CLAVES]):
        js = [c for c in cand if c["fam"] == "jugador"]
        if js:
            return out[:N_CLAVES - 1] + [js[0]]
    return out[:N_CLAVES]


def uso_clave(c):
    """Cuanto recurre a la situacion quien la elige (el ataque, o la defensa si es blitz o caja)."""
    if c["elige"] == "of" and pd.notna(c["x"].get("uso")):
        return f"{c['atk']}: {c['x'].uso * 100:.0f}% de {BASE_OF[c['base']]} ({int(c['x'].rk_uso)}º que más)"
    if c["elige"] == "df" and pd.notna(c["y"].get("uso")):
        return f"{c['dfn']}: {c['y'].uso * 100:.0f}% de {BASE_DEF[c['base']]} ({int(c['y'].rk_uso)}ª que más)"
    return ""


def frase_clave(c):
    """Titular del 2º PNG: solo rangos, que son hechos."""
    return (f"{c['sit']}: {nombre_equipo(c['atk'])}, {int(c['x'].rk)}º en ataque; "
            f"{nombre_equipo(c['dfn'])}, {int(c['y'].rk)}ª en defensa.")


def titular_claves(cs):
    """Solo hay titular si la primera clave es un cruce extremo de verdad
    (top 5 contra las 5 peores, en cualquiera de los dos sentidos)."""
    if not cs or cs[0]["fam"] == "jugador":
        return ""
    ro, rd = int(cs[0]["x"].rk), int(cs[0]["y"].rk)
    return frase_clave(cs[0]) if (ro <= 5 and rd >= 28) or (ro >= 28 and rd <= 5) else ""


def contexto_cob(radar, a, h):
    """Lineas de contexto: cuanto man juega cada defensa y, si es extremo, el slot."""
    cob, cpos = radar["cob"], radar["cpos"]
    if cob is None:
        return []
    out = []
    for t in (a, h):
        if t in cob.index:
            s = (f"Coberturas (Sharp, sem. {radar['sem_cob']}): {t} juega man el "
                 f"{cob.man_pct[t]:.0f}% ({int(cob.rk_man[t])}ª que más)")
            if t in cpos.index and (cpos.rk_slot[t] <= 5 or cpos.rk_slot[t] >= 28):
                s += f" y concede {_com(cpos.ypt_slot[t])} yd por target al slot ({int(cpos.rk_slot[t])}ª)"
            out.append(s)
    return out


def draw_claves(radar, team_a, team_b, cs, out_path, pie="", semana=None):
    """2º PNG de la previa: las claves del partido. Misma cabecera que el 1º;
    cada clave, una tarjeta con el ataque a la izquierda y la defensa a la derecha."""
    fig = lienzo()
    recs = {t: (record_equipo(t), rol) for t, rol in ((team_a, "visitante"), (team_b, "local"))}
    cabecera(fig, (team_a, team_b),
             {t: f"{rec} · {rol}" if rec else rol for t, (rec, rol) in recs.items()})
    antetitulo(fig, f"Claves del partido · semana {semana}" if semana else "Claves del partido",
               y=Y_BAJO_CABECERA)
    tit = titular_claves(cs)
    y_top = (titular(fig, tit, y=0.82, size=18.5) - 0.02) if tit else 0.815
    cob = contexto_cob(radar, team_a, team_b)
    y_fin = 0.088 + 0.015 * len(cob)
    alto = (y_top - y_fin) / max(len(cs), 1)
    for i, c in enumerate(cs):
        y1 = y_top - i * alto
        y0 = y1 - alto + 0.01
        tarjeta(fig, y0, y1)
        yc = (y0 + y1) / 2
        fig.text(0.5, y1 - 0.022, c["sit"].upper(), family=DISP, weight="bold", size=12.5,
                 color=FG, ha="center", va="center")
        for lado, eq, r, x, ha, sg, rol in (("of", c["atk"], c["x"], L + 0.01, "left", 1, "ATAQUE"),
                                            ("df", c["dfn"], c["y"], R - 0.01, "right", -1, "DEFENSA")):
            poner_logo(fig, eq, x + sg * 0.03, yc - 0.004, zoom=0.03)
            rk = int(r.rk)
            fig.text(x + sg * 0.075, yc + 0.002, f"{rk}{'º' if lado == 'of' else 'ª'}", family=DISP,
                     weight="bold", size=24, color=color_rango(rk) if rk <= 5 or rk >= 28 else FG,
                     ha=ha, va="center")
            fig.text(x + sg * 0.17, yc + 0.012, c.get(f"rol_{lado}", f"{rol} {eq}"), family=TXT,
                     weight="bold", size=7.5, color=GRIS1, ha=ha, va="center")
            fig.text(x + sg * 0.17, yc - 0.01, r.txt, family=TXT, size=7.2, color=FG, alpha=0.85,
                     ha=ha, va="center")
        u = uso_clave(c)
        if u:
            fig.text(0.5, y0 + 0.014, u, family=TXT, size=6.8, color=GRIS2, ha="center", va="center",
                     fontstyle="italic")
    for i, linea in enumerate(cob):
        fig.text(0.5, 0.075 + 0.015 * (len(cob) - i), linea, family=TXT, size=6.8, color=GRIS1,
                 ha="center", va="center")
    nota = "Rango entre 32, por EPA por jugada o por la tasa indicada (defensa: 1ª = la mejor)"
    if any(c["fam"] == "jugador" for c in cs):
        nota += " · Receptor: entre los 32 nº 1"
    pie_fn(fig, nota, pie + " · Fuentes: nflverse, " + ", ".join(sorted(radar["fuentes"])))
    guardar(fig, out_path)
    return fig


def linea_clave(c):
    """La clave en una linea, para el TXT del redactor."""
    lado_of = c.get("rol_of", f"ataque {c['atk']}")
    u = uso_clave(c)
    return (f"{c['sit']}: {lado_of} {int(c['x'].rk)}º ({c['x'].txt}) · defensa {c['dfn']} "
            f"{int(c['y'].rk)}ª ({c['y'].txt})" + (f" · {u}" if u else ""))


if __name__ == "__main__":
    from matplotlib.backends.backend_pdf import PdfPages

    # ── Inputs (siempre 4 en orden fijo; los vacíos se ignoran según modo) ────
    modo     = input("¿Partido o Jornada? (p/j): ").strip().lower()
    team_a   = input("Equipo A: ").strip().upper()
    team_b   = input("Equipo B: ").strip().upper()
    week_raw = str(week_cli() or "") or input("Semana: ").strip()

    if modo.startswith("p"):
        week_raw = ""
    else:
        team_a = team_b = ""

    # Semana para archivar el PNG. En modo partido no se pide, asi que se usa
    # hasta donde llegan los datos: basta para que la 2ª previa de un duelo de
    # division no pise a la 1ª, que era lo que pasaba antes.
    week = int(week_raw) if week_raw.isdigit() else None

    # ── Cargar PBP ───────────────────────────────────────────────────────────
    # solo_reg=False: las previas pueden ser de partidos de playoffs
    df, SEASON = cargar_pbp(SEASON, solo_reg=False)
    to_num(df, ["epa","yards_gained","ydstogo","down","return_yards","yardline_100",
                "posteam_score_pre","posteam_score_post","week"])

    # ── Modo semana ───────────────────────────────────────────────────────────
    if week_raw:
        import urllib.request, json as _json

        try:
            week_num = int(week_raw)
        except ValueError:
            raise SystemExit("Semana inválida.")

        # Partidos desde el calendario oficial, que incluye los NO jugados.
        # Detectarlos desde el PBP era una trampa: el sabado por la noche la
        # semana N solo tiene el TNF en el PBP, y el hilo de previas salia
        # con un unico partido en vez de con la jornada entera.
        matchups = []
        dia_partido = {}   # (away, home) -> "Sunday 13:00" (hora ET), para el TXT
        try:
            from pbp_loader import cargar_calendario
            cal, _ = cargar_calendario(SEASON)
            g = cal[(cal["game_type"] == "REG") & (pd.to_numeric(cal["week"], errors="coerce") == week_num)]
            for _, row in g.iterrows():
                away = str(row["away_team"]).strip().upper()
                home = str(row["home_team"]).strip().upper()
                if away not in ("NAN", "") and home not in ("NAN", ""):
                    matchups.append((away, home))
                    dia_partido[(away, home)] = f"{row.get('weekday', '')} {row.get('gametime', '')}".strip()
            if matchups:
                print("  Partidos detectados desde el calendario (schedules).")
        except Exception as e:
            print(f"  Aviso: calendario no disponible ({type(e).__name__}) — probando ESPN")

        # Fallback: ESPN scoreboard API (semanas futuras o sin datos en PBP)
        if not matchups:
            espn_url = (f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
                        f"?seasontype=2&week={week_num}&dates={SEASON}")
            print(f"  Consultando ESPN API para semana {week_num}...")
            try:
                with urllib.request.urlopen(espn_url, timeout=10) as resp:
                    data = _json.loads(resp.read())
                for event in data.get("events", []):
                    comps = event.get("competitions", [{}])[0]
                    home = away = None
                    for c in comps.get("competitors", []):
                        abbr = c.get("team", {}).get("abbreviation", "").upper()
                        if c.get("homeAway") == "home":
                            home = abbr
                        else:
                            away = abbr
                    if home and away:
                        matchups.append((away, home))
            except Exception as e:
                raise SystemExit(f"Error cargando schedule ESPN: {e}")

        if not matchups:
            raise SystemExit(f"No se encontraron partidos para semana {week_num}.")

        print(f"\nPartidos semana {week_num} — {SEASON}:")
        for a, b in matchups:
            print(f"  {a} @ {b}")

        stats_df = df[df["week"] < week_num].copy() if week_num > 1 else df.copy()
        if stats_df.empty:
            print("  Aviso: sin datos previos — usando temporada completa.")
            stats_df = df.copy()
        print(f"  Jugadas para stats: {len(stats_df):,} (semanas 1–{week_num - 1})")

        # El PDF combinado va con sus PNG, en previas/ de la jornada que
        # previsualiza (antes caia suelto en la raiz del repo)
        pdf_path = salida(f"previas_{SEASON}.pdf", SEASON, week_num)
        modo_semana = True

    # ── Modo dos equipos ──────────────────────────────────────────────────────
    else:
        matchups   = [(team_a, team_b)]
        stats_df   = df.copy()
        pdf_path   = None
        modo_semana = False

    # ── Stats y rankings ──────────────────────────────────────────────────────
    off_basic  = compute_offense(stats_df)
    deff_basic = compute_defense(stats_df)
    drv_df     = compute_drive_level(stats_df)
    off_rz, deff_rz = compute_redzone_metrics(drv_df)

    off  = off_basic.join(off_rz,  how="left")
    deff = deff_basic.join(deff_rz, how="left")
    st   = compute_special_teams(stats_df)

    off_high   = ["EPA/jugada","Éxito (%)","EPA/pase","EPA/carrera","Explosivas (%)",
                  "EPA 1º down","EPA downs tardíos","3er down (%)","%RedZone"]
    off_ranks  = rank_dataframe(off,  better_high_cols=off_high,                 better_low_cols=["Distancia media 3º down"])
    deff_ranks = rank_dataframe(deff, better_high_cols=[],                       better_low_cols=list(deff.columns))
    st_ranks   = rank_dataframe(st,   better_high_cols=["FG%","EPA/jugada ST"],  better_low_cols=[])

    # En modo jornada las stats cortan en la semana anterior, que no tiene por
    # que ser la ultima con datos si se regenera una jornada vieja
    if modo_semana and week_num > 1:
        pie = f"NFL {SEASON} · datos hasta sem. {week_num - 1}"
    else:
        pie = sello(SEASON)

    # Conteos de red zone (TD, visitas) para que el % lleve su muestra en el PNG
    _rz = drv_df[drv_df["entered_rz"]]
    rz_cnt = {("of", t): (int(r["sum"]), int(r["size"]))
              for t, r in _rz.groupby("posteam")["td_for"].agg(["sum", "size"]).iterrows()}
    rz_cnt.update({("df", t): (int(r["sum"]), int(r["size"]))
                   for t, r in _rz.groupby("defteam")["td_for"].agg(["sum", "size"]).iterrows()})

    # Lo mismo para el 3er down (convertidos, intentos)
    d3_cnt = {(lado, t): (int(r["hechos"]), int(r["intentos"]))
              for lado, col in (("of", "posteam"), ("df", "defteam"))
              for t, r in tercer_down(stats_df, col).iterrows()}

    # Radar de claves (2º PNG de cada previa). Sin semana, hasta donde llegan los datos
    sem_radar = week_num if modo_semana else int(pd.to_numeric(stats_df["week"]).max()) + 1
    radar = construir_radar(stats_df[stats_df["season_type"] == "REG"], SEASON, sem_radar)

    # ── Generar imágenes ──────────────────────────────────────────────────────
    if modo_semana:
        with PdfPages(pdf_path) as pdf:
            for away, home in matchups:
                out_png = salida(nombre_preview(away, home, SEASON, week_num), SEASON, week_num)
                fig = draw_png(away, home, off, deff, st, off_ranks, deff_ranks, st_ranks, out_png,
                               pie, rz_cnt=rz_cnt, semana=week_num, d3_cnt=d3_cnt)
                # Lienzo fijo, sin bbox tight: descoloca los logos
                pdf.savefig(fig, facecolor="#0f1115")
                plt.close(fig)
                print(f"  → {out_png}")
                out_cl = salida(nombre_preview(away, home, SEASON, week_num, "_claves"), SEASON, week_num)
                fig = draw_claves(radar, away, home, claves_partido(radar, away, home), out_cl, pie,
                                  semana=week_num)
                pdf.savefig(fig, facecolor="#0f1115")
                plt.close(fig)
                print(f"  → {out_cl}")
        print(f"\nPDF combinado: {pdf_path}  ({len(matchups)} partidos)")

        # Los mismos numeros de los PNG en texto, para el redactor del hilo
        # del domingo (claude -p no ejecuta codigo: lee esto). Añade lo que
        # el PNG no ve: cuantas visitas a la red zone hay detras de cada % y
        # cuantos FG detras del FG% (27-sep-2026: el 0% de FG de LA era 0/1)
        rz = drv_df[drv_df["entered_rz"]]
        rz_of = rz.groupby("posteam")["td_for"].agg(["sum", "size"])
        rz_df = rz.groupby("defteam")["td_for"].agg(["sum", "size"])
        fga = stats_df[pd.to_numeric(stats_df.get("field_goal_attempt"), errors="coerce") == 1]
        fg_n = fga.groupby("posteam")["field_goal_result"].agg(
            hechos=lambda s: int((s == "made").sum()), intentos="size")
        pct = {"Éxito (%)", "Explosivas (%)", "%RedZone", "Éxito permitido (%)",
               "3er down (%)", "3er down permitido (%)",
               "Explosivas permitidas (%)", "%RedZone permitido", "FG%"}
        lineas = [f"NUMEROS DE LAS PREVIAS — semana {week_num} de {SEASON}",
                  f"Stats de las semanas 1-{week_num - 1}; rango 1 = el mejor de 32 "
                  "(en defensa, 1 = la que menos concede). Mismos numeros que los PNG.",
                  "Orden: kickoff (el NN_ del PNG). Dia y hora ET del calendario oficial:",
                  "el hilo del domingo lleva SOLO los de Sunday; los de Monday van el lunes.", ""]
        for n, (away, home) in enumerate(matchups, 1):
            dia = dia_partido.get((away, home), "") if modo_semana else ""
            lineas.append(f"== {n:02d} {away} @ {home} — {dia or 'dia ?'} — "
                          f"{os.path.basename(salida(nombre_preview(away, home, SEASON, week_num), SEASON, week_num))}")
            _d = mayor_desajuste(away, home, off_ranks, deff_ranks, {k: v[1] for k, v in rz_cnt.items()})
            lineas.append("  TITULAR DEL PNG: " + (frase_titular(_d) if _d else "(sin titular: ningun cruce extremo)"))
            for t in (away, home):
                lineas.append(f"  {t} ({record_equipo(t) or '?'})")
                for titulo, tabla, rangos in (("ATAQUE", off, off_ranks),
                                              ("DEFENSA", deff, deff_ranks),
                                              ("ESPECIALES", st, st_ranks)):
                    celdas = [f"{c} {fmt_val(tabla.loc[t, c], c in pct)} ({int(rangos.loc[t, c])}º)"
                              for c in tabla.columns if t in tabla.index]
                    lineas.append(f"    {titulo}: " + " · ".join(celdas))
                ro = rz_of.loc[t] if t in rz_of.index else None
                rd = rz_df.loc[t] if t in rz_df.index else None
                fg = fg_n.loc[t] if t in fg_n.index else None
                lineas.append(
                    "    CONTEO: red zone ataque "
                    + (f"{int(ro['sum'])} TD en {int(ro['size'])} visitas" if ro is not None else "sin visitas")
                    + " · red zone rival "
                    + (f"{int(rd['sum'])} TD en {int(rd['size'])} visitas" if rd is not None else "sin visitas")
                    + " · FG "
                    + (f"{fg['hechos']}/{fg['intentos']}" if fg is not None else "0/0"))
            _cs = claves_partido(radar, away, home)
            lineas.append("  CLAVES (2º PNG, "
                          + os.path.basename(salida(nombre_preview(away, home, SEASON, week_num, "_claves"),
                                                    SEASON, week_num)) + "):")
            lineas.append("    TITULAR DEL PNG: " + (titular_claves(_cs) or "(sin titular: ningun cruce extremo)"))
            lineas += [f"    {i}. {linea_clave(c)}" for i, c in enumerate(_cs, 1)]
            lineas += [f"    {x}" for x in contexto_cob(radar, away, home)]
            lineas.append("")
        txt_path = salida(f"previas_numeros_{SEASON}.txt", SEASON, week_num)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lineas))
        print(f"Numeros para el hilo: {txt_path}")
    else:
        away, home = matchups[0]
        out_png = salida(nombre_preview(away, home, SEASON, week), SEASON, week)
        fig = draw_png(away, home, off, deff, st, off_ranks, deff_ranks, st_ranks, out_png, pie,
                       rz_cnt=rz_cnt, semana=week, d3_cnt=d3_cnt)
        plt.close(fig)
        print(f"\nPNG generado: {out_png}")
        out_cl = salida(nombre_preview(away, home, SEASON, week, "_claves"), SEASON, week)
        plt.close(draw_claves(radar, away, home, claves_partido(radar, away, home), out_cl, pie,
                              semana=week))
        print(f"PNG de claves: {out_cl}")
