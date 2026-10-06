# DatoSemana.py
# El Dato de la Semana: lo más raro que pasó en la jornada, buscado en MUCHAS
# situaciones y comparado con la jornada Y con la historia.
#
# Rehecho el 06-oct-2026 (Luis: "es muy poco espectro... si algún equipo
# destaca un día porque con formación de tres TE generó el mayor EPA histórico,
# quiero saberlo"). Antes miraba 5 métricas de equipo contra la jornada.
#
# Candidatos: cada ataque en cada SITUACIÓN de su partido (EPA por jugada):
#   equipo      todas las jugadas, pase, carrera
#   situación   3er down, red zone
#   formación   shotgun, bajo centro, sin huddle                 (PBP)
#   charting    play action, motion, pantallas, backfield vacío,
#               contra el blitz, carrera contra caja de 8+ / de 6- (FTN, 2022+)
#   personal    11, 12, 13, 21, 22                               (Sharp, 2021+)
#   especiales  EPA por jugada; % de field goals (solo contra la jornada)
# Cada uno se mide de dos formas, encogidas por el tamaño de la muestra:
#   - contra la jornada: z robusto entre los equipos de esa semana
#   - contra la historia: su puesto entre TODOS los partidos de equipo de esa
#     situación desde el primer año con datos (PBP 2016, FTN 2022, Sharp 2021)
# Gana el más raro contra la historia; la semana es contexto. Un ataque hundido se cuenta desde la defensa
# rival, que es la mejor historia. El TXT trae los 8 primeros para elegir y el
# PNG sale del 1º (o del que se pida con --elegir N).
#
# El charting de FTN llega partido a partido y tarda días: el dato espera a que
# estén todos los partidos MENOS el Monday Night, que FTN carga el miércoles
# (decisión de Luis, 06-oct-2026: sale el martes; el jueves no cabe). Sin eso
# el script se para; con --sin-ftn sale igual sin las situaciones de charting.
#
#   python DatoSemana.py --week 4
#   python DatoSemana.py --week 4 --elegir 3      # el 3º candidato del TXT
#   python DatoSemana.py --week 4 --excluir MIN   # MIN no puede ser el dato

import argparse
import os
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

from pbp_loader import (cargar_pbp, cargar_ftn, cargar_calendario, salida, season_cli,
                        week_cli, ultima_semana, temporada_actual, DatosNoDisponibles)
from estilo import *   # noqa: F401,F403  (estilo comun, docs/estilo-visual.md)

sys.stdout.reconfigure(encoding="utf-8")

DESDE_PBP, DESDE_FTN, DESDE_SHARP = 2016, 2022, 2021
K_MUESTRA = 15          # encogimiento: n / (n + 15)
MIN_EQUIPOS_SEMANA = 16 # con menos equipos, la comparación con la semana engaña (13 personal: 9)
TOP_HIST = 10           # dentro del top 10 histórico, el titular cuenta la historia
TOP_TXT = 8

# clave: (familia, base, mínimo de jugadas, contexto para la frase, etiqueta corta)
SITUACIONES = {
    "todas":  ("equipo", "all", 25, "", "EPA por jugada"),
    "pase":   ("equipo", "db", 18, "pasando", "EPA por pase"),
    "carrera": ("equipo", "run", 12, "corriendo", "EPA por carrera"),
    "d3":     ("situación", "all", 8, "en 3er down", "EPA en 3er down"),
    "rz":     ("situación", "all", 8, "en la red zone", "EPA en red zone"),
    "sg":     ("formación", "all", 15, "desde shotgun", "EPA desde shotgun"),
    "uc":     ("formación", "all", 10, "con el QB bajo centro", "EPA bajo centro"),
    "nh":     ("formación", "all", 8, "sin huddle", "EPA sin huddle"),
    "pa":     ("charting", "db", 8, "con play action", "EPA con play action"),
    "mot":    ("charting", "all", 15, "con motion", "EPA con motion"),
    "scr":    ("charting", "db", 6, "con pantallas", "EPA en pantallas"),
    "empty":  ("charting", "db", 8, "con el backfield vacío", "EPA con backfield vacío"),
    "blitz":  ("charting", "db", 8, "contra el blitz", "EPA contra el blitz"),
    "box8":   ("charting", "run", 6, "corriendo contra 8 o más en la caja", "EPA contra caja de 8+"),
    "box6":   ("charting", "run", 8, "corriendo contra 6 o menos en la caja", "EPA contra caja de 6-"),
    "p11":    ("personal", "sharp", 15, "con tres receptores (11 personal)", "EPA con 11 personal"),
    "p12":    ("personal", "sharp", 10, "con dos TE (12 personal)", "EPA con 12 personal"),
    "p13":    ("personal", "sharp", 8, "con tres TE (13 personal)", "EPA con 13 personal"),
    "p21":    ("personal", "sharp", 8, "con dos RB (21 personal)", "EPA con 21 personal"),
    "p22":    ("personal", "sharp", 6, "con dos RB y dos TE (22 personal)", "EPA con 22 personal"),
    "st":     ("especiales", "st", 10, "en equipos especiales", "EPA en especiales"),
}
DESDE = {"charting": DESDE_FTN, "personal": DESDE_SHARP}


def args_cli():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--excluir", default="")
    ap.add_argument("--elegir", type=int, default=1)
    ap.add_argument("--sin-ftn", action="store_true")
    a, _ = ap.parse_known_args()
    a.excluir = {x.strip().upper() for x in a.excluir.split(",") if x.strip()}
    return a


# ── Datos ─────────────────────────────────────────────────────────────────────
COLS = ["season", "week", "game_id", "play_id", "posteam", "defteam", "play_type", "epa",
        "success", "qb_dropback", "qb_scramble", "qb_kneel", "qb_spike", "down",
        "yardline_100", "shotgun", "no_huddle", "field_goal_result", "season_type"]


def jugadas_pbp(season):
    df, _ = cargar_pbp(season, columns=COLS, avisar=False)
    return df[df.season_type == "REG"] if "season_type" in df.columns else df


def celdas_pbp(df, ftn=None):
    """Una fila por (partido, ataque, situación) con n y EPA medio."""
    st_tipos = {"kickoff", "punt", "field_goal", "extra_point"}
    st = df[df.play_type.isin(st_tipos) & df.epa.notna() & df.posteam.notna()]
    p = df[df.play_type.isin(["pass", "run"]) & df.epa.notna() & df.posteam.notna()
           & (df.qb_kneel != 1) & (df.qb_spike != 1)].copy()
    p["db"] = p.qb_dropback == 1
    p["run"] = (p.play_type == "run") & (p.qb_scramble != 1)
    p["all"] = True
    m = {"todas": p["all"], "pase": p.db, "carrera": p.run, "d3": p.down == 3,
         "rz": p.yardline_100 <= 20, "sg": p.shotgun == 1, "uc": p.shotgun == 0,
         "nh": p.no_huddle == 1}
    if ftn is not None and len(ftn):
        f = ftn[ftn.qb_location.astype(str) != "0"]
        p = p.merge(f[["nflverse_game_id", "nflverse_play_id", "is_play_action", "is_motion",
                       "is_screen_pass", "n_offense_backfield", "n_blitzers", "n_defense_box"]],
                    left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"],
                    how="left")
        hay = p.nflverse_play_id.notna()
        box = pd.to_numeric(p.n_defense_box, errors="coerce")
        m = {k: v.reindex(p.index) if hasattr(v, "reindex") else v for k, v in
             {"todas": p["all"], "pase": p.db, "carrera": p.run, "d3": p.down == 3,
              "rz": p.yardline_100 <= 20, "sg": p.shotgun == 1, "uc": p.shotgun == 0,
              "nh": p.no_huddle == 1}.items()}
        m.update({"pa": hay & p.db & (p.is_play_action == True),            # noqa: E712
                  "mot": hay & (p.is_motion == True),                        # noqa: E712
                  "scr": hay & p.db & (p.is_screen_pass == True),            # noqa: E712
                  "empty": hay & p.db & (pd.to_numeric(p.n_offense_backfield, errors="coerce") == 0),
                  "blitz": hay & p.db & (pd.to_numeric(p.n_blitzers, errors="coerce") >= 1),
                  "box8": hay & p.run & (box >= 8), "box6": hay & p.run & (box <= 6)})
    trozos = []
    for k, mask in m.items():
        g = (p[mask.fillna(False).astype(bool)]
             .groupby(["season", "week", "game_id", "posteam", "defteam"]).epa
             .agg(["mean", "size"]).reset_index())
        g["sit"] = k
        trozos.append(g)
    g = st.groupby(["season", "week", "game_id", "posteam", "defteam"]).epa.agg(["mean", "size"]).reset_index()
    g["sit"] = "st"
    trozos.append(g)
    return pd.concat(trozos, ignore_index=True).rename(columns={"mean": "epa", "size": "n"})


def celdas_sharp(season):
    """Personal ofensivo por partido desde el fichero de Sharp. Las temporadas
    cerradas se guardan en pbp_cache/ (no cambian)."""
    from copiar_personal_web import sharp_jugadas
    ruta = os.path.join("pbp_cache", f"sharp_personal_{season}.parquet")
    if season < temporada_actual() and os.path.exists(ruta):
        pl = pd.read_parquet(ruta)
    else:
        pl, _ = sharp_jugadas(season)
        if season < temporada_actual():
            pl.to_parquet(ruta)
    pl = pl[pl.week <= 18]
    cal, _ = cargar_calendario(season)
    cal = cal[cal.game_type == "REG"]
    riv, gid = {}, {}
    for r in cal.itertuples():
        for a, b in ((r.home_team, r.away_team), (r.away_team, r.home_team)):
            riv[(int(r.week), a)] = b
            gid[(int(r.week), a)] = r.game_id
    pl = pl[pl.grupo.isin([11, 12, 13, 21, 22])].copy()
    pl["defteam"] = [riv.get((w, t)) for w, t in zip(pl.week, pl.team)]
    pl["game_id"] = [gid.get((w, t)) for w, t in zip(pl.week, pl.team)]
    g = (pl.groupby(["week", "game_id", "team", "defteam", "grupo"]).epa.agg(["mean", "size"])
           .reset_index().rename(columns={"team": "posteam", "mean": "epa", "size": "n"}))
    g["sit"] = "p" + g.grupo.astype(str)
    g["season"] = season
    return g.drop(columns="grupo")


def ftn_completo(season, week, pbp_w):
    """(¿listo?, partidos con charting, partidos jugados, texto). Listo = todos
    menos, como mucho, el Monday Night: FTN carga los del domingo el martes
    hacia las 13:00 y el MNF el miércoles por la noche, y el dato sale el
    martes (Luis, 06-oct-2026: el jueves no cabe)."""
    jugados = set(pbp_w.game_id.unique())
    try:
        f, _ = cargar_ftn(season)
    except DatosNoDisponibles:
        return False, 0, len(jugados), "sin FTN"
    hechos = set(f[f.week == week].nflverse_game_id.unique()) & jugados
    faltan = jugados - hechos
    cal, _ = cargar_calendario(season)
    mnf = set(cal[(cal.week == week) & (cal.weekday == "Monday")].game_id)
    listo = faltan <= mnf
    nota_ = "completo" if not faltan else (
        f"sin el Monday Night ({', '.join(sorted(faltan))})" if listo else f"faltan {len(faltan)}")
    return listo, len(hechos), len(jugados), nota_


# ── Rareza ────────────────────────────────────────────────────────────────────
def encoger(n):
    return np.sqrt(n / (n + K_MUESTRA))


def z_robusto(s):
    med = s.median()
    mad = (s - med).abs().median()
    if mad > 0:
        return 0.67448975 * (s - med) / mad
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0


def puntuar(hist, season, week):
    """Candidatos de la jornada con su rareza contra la semana y la historia."""
    nd = NormalDist()
    out = []
    for sit, (fam, _, minimo, _, _) in SITUACIONES.items():
        h = hist[(hist.sit == sit) & (hist.n >= minimo)]
        desde = DESDE.get(fam, DESDE_PBP)
        h = h[h.season >= desde]
        w = h[(h.season == season) & (h.week == week)].copy()
        if len(w) < 3:
            continue
        w["zw"] = (z_robusto(w.epa) * encoger(w.n)) if len(w) >= MIN_EQUIPOS_SEMANA else 0.0
        orden = h.epa.rank(method="min")                      # 1 = el peor
        N = len(h)
        w["rk_hist_mejor"] = (N - orden.loc[w.index] + 1).astype(int)
        w["rk_hist_peor"] = orden.loc[w.index].astype(int)
        p = ((orden.loc[w.index] - 0.5) / N).clip(1e-4, 1 - 1e-4)
        w["zh"] = [nd.inv_cdf(x) for x in p] * encoger(w.n)
        w["N_hist"] = N
        w["desde"] = desde
        w["rk_sem_mejor"] = w.epa.rank(ascending=False, method="min").astype(int)
        w["rk_sem_peor"] = w.epa.rank(method="min").astype(int)
        w["n_sem"] = len(w)
        # Se ordena por la HISTORIA (que ya incluye la semana): con 31 equipos el
        # z de la semana exagera (06-oct-2026: el 69º de 5.383 salía 1º). La
        # semana queda como contexto y para el titular
        w["score"] = w.zh
        out.append(w)
    return pd.concat(out, ignore_index=True)


# ── Frases ────────────────────────────────────────────────────────────────────
def frase(c):
    """Titular: solo hechos que el script calcula (puestos y valores)."""
    _, _, _, ctx, _ = SITUACIONES[c.sit]
    ctx = (" " + ctx) if ctx else ""
    A, D = nombre_equipo(c.posteam), nombre_equipo(c.defteam)
    val = num(c.epa)
    bueno = c.score > 0
    if c.sit == "st":
        s = (f"Ningunos especiales sumaron tanto esta semana como los de {A}: {val} de EPA por jugada."
             if bueno else
             f"Ningunos especiales costaron tanto esta semana como los de {A}: {val} de EPA por jugada.")
        return s[0].upper() + s[1:]
    rk_h = c.rk_hist_mejor if bueno else c.rk_hist_peor
    historico = rk_h <= TOP_HIST or abs(c.zh) >= abs(c.zw)
    if bueno:
        if historico and c.rk_hist_mejor == 1:
            s = (f"Desde {c.desde}, ningún ataque había rendido tanto{ctx} en un partido: "
                 f"{A}, {val} de EPA por jugada.")
        elif not historico and c.rk_sem_mejor == 1:
            s = f"Ningún ataque rindió tanto{ctx} esta semana como {A}: {val} de EPA por jugada."
        else:
            s = (f"{A} {v(A, 'sumó', 'sumaron')} {val} de EPA por jugada{ctx}: "
                 + (f"el {c.rk_hist_mejor}º mejor partido de {miles(c.N_hist)} desde {c.desde}."
                    if c.rk_hist_mejor <= TOP_HIST else
                    f"mejor que el {pct_mejor(c.rk_hist_mejor, c.N_hist)} de los partidos desde {c.desde}."))
    else:
        if historico and c.rk_hist_peor == 1:
            s = (f"Desde {c.desde}, ninguna defensa había frenado tanto a un ataque{ctx}: "
                 f"{D} {v(D, 'dejó', 'dejaron')} a {A} en {val} de EPA por jugada.")
        elif not historico and c.rk_sem_peor == 1:
            s = (f"Ninguna defensa frenó tanto a un ataque{ctx} esta semana como la de {D}: "
                 f"{v(D, 'dejó', 'dejaron')} a {A} en {val} de EPA por jugada.")
        else:
            s = (f"{D} {v(D, 'dejó', 'dejaron')} a {A} en {val} de EPA por jugada{ctx}: "
                 + (f"el {c.rk_hist_peor}º peor partido de {miles(c.N_hist)} desde {c.desde}."
                    if c.rk_hist_peor <= TOP_HIST else
                    f"peor que el {pct_mejor(c.rk_hist_peor, c.N_hist)} de los partidos desde {c.desde}."))
    return s[0].upper() + s[1:]


def pct_mejor(rk, N):
    """Puesto rk de N -> 'del 97%' (porcentaje de partidos que quedan por detrás)."""
    return f"{int((N - rk) / N * 100)}%"


def miles(n):
    return f"{int(n):,}".replace(",", ".")


def linea_txt(i, c):
    fam = SITUACIONES[c.sit][0]
    lado = ("especiales" if c.sit == "st" else
            "ataque" if c.score > 0 else f"defensa de {c.defteam}")
    return (f"{i}. [{fam}] {c.posteam} vs {c.defteam} · {SITUACIONES[c.sit][4]}: {c.epa:+.2f} en "
            f"{int(c.n)} jugadas ({lado})\n"
            f"     semana: {c.rk_sem_mejor if c.score > 0 else c.rk_sem_peor}º de {c.n_sem} "
            f"(z {c.zw:+.2f}) · historia: {c.rk_hist_mejor if c.score > 0 else c.rk_hist_peor}º de "
            f"{miles(c.N_hist)} desde {c.desde} (z {c.zh:+.2f})\n"
            f"     Titular: {frase(c)}")


# ── PNG ───────────────────────────────────────────────────────────────────────
def dibujar(c, semana_cells, season, week, outfile):
    bueno = c.score > 0
    prota = c.posteam if bueno else c.defteam
    s = semana_cells.set_index("posteam" if bueno else "defteam").epa
    s = s.sort_values(ascending=not bueno)                    # el protagonista arriba si es el 1º
    teams = s.index.tolist()
    n = len(teams)
    rk = teams.index(prota) + 1
    fig = lienzo()
    antetitulo(fig, f"Dato de la semana {week}")
    y_top = titular(fig, frase(c), y=0.925, size=21) - 0.03
    yc = y_top - 0.04
    poner_logo(fig, prota, L + 0.035, yc, zoom=0.024)
    fig.text(L + 0.085, yc, prota, family=TXT, weight="bold", size=9, color=color_equipo(prota),
             va="center")
    lado = ("ATAQUE" if bueno else f"DEFENSA · ante {c.posteam}") if c.sit != "st" else "ESPECIALES"
    fig.text(L + 0.15, yc, f"·  {lado}", family=TXT, weight="semibold", size=8.5, color=GRIS1,
             va="center")
    etiqueta(fig, SITUACIONES[c.sit][4] + ("" if bueno else " permitido"), L + 0.01, yc - 0.05)
    numero(fig, num(c.epa), L + 0.005, yc - 0.11)
    rk_h = c.rk_hist_mejor if bueno else c.rk_hist_peor
    fig.text(L + 0.43, yc - 0.083, f"{rk}º de {n} esta semana", family=TXT, weight="bold", size=9,
             color=color_rango(rk, n), va="center")
    fig.text(L + 0.43, yc - 0.108, f"{rk_h}º de {miles(c.N_hist)} partidos desde {c.desde}",
             family=TXT, weight="bold", size=8.5, color=FG if rk_h <= 10 else GRIS1, va="center")
    nota(fig, f"{int(c.n)} jugadas", L + 0.43, yc - 0.132)
    etiqueta(fig, f"Los {n} equipos de la semana", L + 0.01, yc - 0.19)
    ax = estilo_ejes(fig.add_axes([L + 0.01, 0.17, R - L - 0.02, yc - 0.215 - 0.17]), x=True)
    vals = s.values
    otro = teams[1] if rk == 1 else teams[0]
    for i, (t, val) in enumerate(zip(teams, vals)):
        yy = n - 1 - i
        es = t == prota
        col = color_equipo(t) if es else (GRIS1 if t == otro else NEUTRO)
        ax.plot([0, val], [yy, yy], color=col, lw=1.6 if es else 1.0, solid_capstyle="butt")
        ax.scatter([val], [yy], s=26 if es else 9, color=col, zorder=3, lw=0)
        if es or t == otro:
            ax.text(val + (0.012 if val >= 0 else -0.012) * (max(abs(vals)) or 1), yy, t,
                    family=TXT, weight="bold" if es else "normal", size=7, color=col,
                    va="center", ha="left" if val >= 0 else "right")
    ax.axvline(0, color=GRID, lw=1)
    span = max(abs(vals)) or 1
    ax.set_xlim(min(0, vals.min()) - 0.15 * span, max(0, vals.max()) + 0.15 * span)
    ax.set_ylim(-1, n)
    ax.set_xticks([])
    izq, der = ("PEOR", "MEJOR") if bueno else ("MEJOR", "PEOR")
    if not bueno:
        izq, der = "MEJOR DEFENSA", "PEOR DEFENSA"
    fig.text(L + 0.01, 0.155, izq, family=TXT, weight="medium", size=6.5, color=GRIS2, va="center")
    fig.text(R - 0.01, 0.155, der, family=TXT, weight="medium", size=6.5, color=GRIS2, va="center",
             ha="right")
    tarjeta(fig, 0.13, y_top)
    fuente = {"charting": "nflverse PBP + charting de FTN", "personal": "personal de Sharp Football"}
    pie(fig, f"{SITUACIONES[c.sit][4]} · mínimo {SITUACIONES[c.sit][2]} jugadas · historia: todos los "
             f"partidos de equipo desde {c.desde}",
        f"NFL {season} · semana {week} · {fuente.get(SITUACIONES[c.sit][0], 'nflverse PBP')}")
    guardar(fig, outfile)


# ── FG% (solo contra la jornada, como antes) ──────────────────────────────────
def candidato_fg(dfw):
    fg = dfw[(dfw.play_type == "field_goal") & dfw.posteam.notna()]
    if fg.empty:
        return None
    made = fg.field_goal_result.astype(str).str.lower().eq("made")
    g = made.groupby(fg.posteam).agg(["mean", "size"])
    g = g[g["size"] >= 3]
    if len(g) < 6:
        return None
    z = z_robusto(g["mean"]) * encoger(g["size"])
    t = z.abs().idxmax()
    return t, float(g.loc[t, "mean"]), int(g.loc[t, "size"]), float(z[t])


def main():
    a = args_cli()
    season = season_cli() or temporada_actual()
    week = week_cli() or ultima_semana(season)
    print(f"Dato de la semana {week} — {season}")

    actual = jugadas_pbp(season)
    dfw = actual[actual.week == week]
    if dfw.empty:
        raise SystemExit(f"No hay jugadas para la semana {week}.")
    completo, hechos, jugados, nota_ftn = ftn_completo(season, week, dfw)
    print(f"Charting de FTN: {hechos} de {jugados} partidos de la semana {week} ({nota_ftn})")
    if not completo and not a.sin_ftn:
        raise SystemExit("  FTN aún no está completo: el dato espera (decisión del 06-oct-2026). "
                         "Para sacarlo igual, --sin-ftn.")

    trozos = []
    for s in range(DESDE_PBP, season + 1):
        df = actual if s == season else jugadas_pbp(s)
        ftn = None
        if s >= DESDE_FTN and (completo or s < season):
            try:
                ftn, _ = cargar_ftn(s)
            except DatosNoDisponibles:
                ftn = None
        trozos.append(celdas_pbp(df, ftn))
    for s in range(DESDE_SHARP, season + 1):
        try:
            trozos.append(celdas_sharp(s))
        except Exception as e:      # noqa: BLE001 — sin Sharp, sin personal
            print(f"  Aviso: sin personal de Sharp {s} ({type(e).__name__}: {e})")
    hist = pd.concat(trozos, ignore_index=True)
    hist = hist[hist.posteam.notna() & hist.defteam.notna()]

    cand = puntuar(hist, season, week)
    prota = np.where(cand.score > 0, cand.posteam, cand.defteam)
    cand = cand[~pd.Series(prota).isin(a.excluir).values]
    cand = cand.reindex(cand.score.abs().sort_values(ascending=False).index).reset_index(drop=True)
    # Un solo candidato por equipo y situación de familia parecida: fuera los
    # duplicados exactos (mismo equipo, mismo partido, mismo lado)
    cand["clave"] = [f"{p}-{d}-{'+' if sc > 0 else '-'}" for p, d, sc in
                     zip(cand.posteam, cand.defteam, cand.score)]
    top = cand.drop_duplicates("clave").head(TOP_TXT)

    print(f"\nCANDIDATOS (los {len(top)} más raros; elige con --elegir N):")
    for i, c in enumerate(top.itertuples(), 1):
        print(linea_txt(i, c))
    fg = candidato_fg(dfw)
    if fg:
        print(f"\n  Field goals (solo contra la jornada): {fg[0]} {fg[1] * 100:.0f}% en {fg[2]} "
              f"intentos (z {fg[3]:+.2f})")

    elegido = top.iloc[a.elegir - 1]
    if elegido.score > 0:
        print(f"\nOutlier -> {elegido.posteam} en '{SITUACIONES[elegido.sit][4]}' (z={elegido.score:+.2f})")
    else:
        print(f"\nOutlier -> {elegido.posteam} hundido en '{SITUACIONES[elegido.sit][4]}' "
              f"(z={elegido.score:+.2f})")
        print(f"Reencuadrado como defensa: {elegido.defteam} en '{SITUACIONES[elegido.sit][4]} permitido'")
    sem = hist[(hist.season == season) & (hist.week == week) & (hist.sit == elegido.sit)
               & (hist.n >= SITUACIONES[elegido.sit][2])]
    outfile = salida(f"dato_semana_outlier_{season}.png", season, week)
    dibujar(elegido, sem, season, week, outfile)
    print(f"Titular: {frase(elegido)}")
    print(f"Guardado: {outfile}")


if __name__ == "__main__":
    main()
