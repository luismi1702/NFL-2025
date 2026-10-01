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
                    FG, GRIS1)
from matplotlib.patches import FancyBboxPatch
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
    ("%RedZone", "%RedZone permitido", "TD en red zone", "rz",
     "en TD en la red zone", "defendiendo su red zone"),
]
# Para el titular se prefieren los cruces concretos a los generales
PREFERENCIA = ["EPA/pase", "EPA/carrera", "Explosivas (%)", "%RedZone", "EPA/jugada", "Éxito (%)"]
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


def valor_celda(v, formato, team=None, lado=None, rz_cnt=None):
    if pd.isna(v):
        return "–"
    if formato == "num":
        return num(v)
    if formato == "rz" and rz_cnt:
        hechos, visitas = rz_cnt.get((lado, team), (None, None))
        if visitas:
            return f"{v:.0f}% ({hechos}/{visitas})"
    return f"{v:.0f}%"


def draw_png(team_a, team_b, off, deff, st, off_r, deff_r, st_r, out_path, pie="",
             rz_cnt=None, semana=None):
    """Previa de un partido (team_a visita a team_b), estilo docs/estilo-visual.md:
    titular con el mayor desajuste y los dos cruces ataque contra defensa."""
    rz_cnt = rz_cnt or {}
    rz_n = {k: v[1] for k, v in rz_cnt.items()}
    COLP = colores_pareja(team_a, team_b)
    fig = lienzo()
    antetitulo(fig, f"Previa · semana {semana}" if semana else "Previa")
    d = mayor_desajuste(team_a, team_b, off_r, deff_r, rz_n)
    y_top = (titular(fig, frase_titular(d), y=0.925) - 0.03) if d else 0.90
    destacado = (d[0], d[2][0]) if d else None

    # Cabecera de equipos (la tarjeta se dibuja al final, ajustada al contenido)
    yc = y_top - 0.045
    for x, t, rol in ((L + 0.035, team_a, "visitante"), (0.535, team_b, "local")):
        poner_logo(fig, t, x, yc, zoom=0.022)
        fig.text(x + 0.045, yc + 0.008, t, family=TXT, weight="bold", size=10,
                 color=COLP[t], va="center")
        rec = record_equipo(t)
        fig.text(x + 0.045, yc - 0.017, f"{rec} · {rol}" if rec else rol, family=TXT, size=7.5,
                 color=GRIS1, va="center")

    # Dos cruces: cuando ataca el visitante y cuando ataca el local
    y = yc - 0.06
    alto_fila = 0.030
    for atk, dfn in ((team_a, team_b), (team_b, team_a)):
        linea_h(fig, y + 0.018)
        etiqueta(fig, f"Cuando ataca {atk}", L + 0.01, y)
        fig.text(0.60, y, f"{atk} ATAQUE", family=TXT, weight="semibold", size=6.8,
                 color=COLP[atk], va="center", ha="right")
        fig.text(R - 0.005, y, f"{dfn} DEFENSA", family=TXT, weight="semibold", size=6.8,
                 color=COLP[dfn], va="center", ha="right")
        y -= 0.034
        for co, cd, lab, formato, _, _ in CRUCES:
            vo = off.loc[atk, co] if (atk in off.index and co in off.columns) else np.nan
            vd = deff.loc[dfn, cd] if (dfn in deff.index and cd in deff.columns) else np.nan
            ro = off_r.loc[atk, co] if (atk in off_r.index and co in off_r.columns) else np.nan
            rd = deff_r.loc[dfn, cd] if (dfn in deff_r.index and cd in deff_r.columns) else np.nan
            if destacado == (atk, co):
                fig.add_artist(FancyBboxPatch((L, y - alto_fila / 2), R - L, alto_fila,
                                              boxstyle="round,pad=0,rounding_size=0.006",
                                              transform=fig.transFigure, fc="#1e2433",
                                              ec="none", zorder=-0.5))
            fig.text(L + 0.01, y, lab, family=TXT, size=8.5, color=FG, va="center")
            for xv, v, rk, team, lado in ((0.60, vo, ro, atk, "of"),
                                          (R - 0.005, vd, rd, dfn, "df")):
                fig.text(xv - 0.075, y, valor_celda(v, formato, team, lado, rz_cnt), family=TXT,
                         weight="semibold", size=8.5, color=FG, va="center", ha="right")
                if pd.notna(rk):
                    fig.text(xv, y, f"{int(rk)}º", family=TXT, weight="bold", size=8.5,
                             color=color_rango(int(rk)), va="center", ha="right")
            y -= alto_fila
        y -= 0.022

    # Especiales en una linea
    linea_h(fig, y + 0.012)

    def esp(t):
        fg = st.loc[t, "FG%"] if t in st.index else np.nan
        rk = st_r.loc[t, "FG%"] if t in st_r.index else np.nan
        return f"{t} {fg:.0f}% ({int(rk)}º)" if pd.notna(fg) and pd.notna(rk) else f"{t} –"
    etiqueta(fig, "Field goals", L + 0.01, y - 0.012)
    fig.text(R - 0.005, y - 0.012, f"{esp(team_a)}   ·   {esp(team_b)}", family=TXT, size=8,
             color=GRIS1, va="center", ha="right")
    tarjeta(fig, y - 0.045, y_top)

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
                  "EPA 1º down","EPA downs tardíos","%RedZone"]
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

    # ── Generar imágenes ──────────────────────────────────────────────────────
    if modo_semana:
        with PdfPages(pdf_path) as pdf:
            for away, home in matchups:
                out_png = salida(nombre_preview(away, home, SEASON, week_num), SEASON, week_num)
                fig = draw_png(away, home, off, deff, st, off_ranks, deff_ranks, st_ranks, out_png,
                               pie, rz_cnt=rz_cnt, semana=week_num)
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
                       rz_cnt=rz_cnt, semana=week)
        plt.close(fig)
        print(f"\nPNG generado: {out_png}")
