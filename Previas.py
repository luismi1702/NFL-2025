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
                    FG, GRIS1, GRIS2, DISP)
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# === Config ===
SEASON       = season_cli()   # None = auto-detectar última temporada
EXP_PASS_YDS = 15   # jugada explosiva pase
EXP_RUN_YDS  = 10   # jugada explosiva carrera
FIGSIZE      = (8.7, 13.5)
DPI          = 170

# ---------------- Nombre del PNG ----------------
def nombre_preview(away, home, season, week):
    """preview_... con el indice de kickoff delante (01 = TNF), como las fichas
    y los resumenes, para que la carpeta de la semana se lea en orden de partido."""
    orden = orden_partido(season, week, away, home) if week else None
    if orden:
        idx, vis, loc = orden
        return f"{idx:02d}_preview_{vis}_vs_{loc}_{season}.png"
    return f"preview_{away}_vs_{home}_{season}.png"

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
    dsub = df[df["posteam"].notna()].copy()
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
    COLP = colores_pareja(team_a, team_b)
    XM = 0.5
    fig = lienzo()

    # Cabecera: cada mitad teñida con el color de su equipo, logo, siglas y record
    alto_cab = 0.15
    for t, x0, rol in ((team_a, 0.0, "visitante"), (team_b, 0.5, "local")):
        ax = fig.add_axes([x0, 1 - alto_cab, 0.5, alto_cab], zorder=-2)
        g = np.zeros((60, 2, 4))
        g[..., :3] = to_rgb(COLP[t])
        g[..., 3] = np.linspace(0.50, 0.0, 60)[:, None]
        ax.imshow(g, aspect="auto", extent=(0, 1, 0, 1))
        ax.axis("off")
        poner_logo(fig, t, x0 + 0.16, 0.925, zoom=0.07)
        fig.text(x0 + 0.27, 0.935, t, family=DISP, weight="bold", size=20, color=FG, va="center")
        rec = record_equipo(t)
        fig.text(x0 + 0.27, 0.902, f"{rec} · {rol}" if rec else rol, family=TXT, size=7.5,
                 color=FG, alpha=0.75, va="center")
    fig.text(XM, 0.925, "@", family=TXT, weight="bold", size=9, color=GRIS1, ha="center",
             va="center")

    antetitulo(fig, f"Previa · semana {semana}" if semana else "Previa", y=0.84)
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
