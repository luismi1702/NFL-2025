"""
discriminacion_total.py
Analisis discriminante completo: 35+ metricas para identificar cuales
realmente diferencian a los campeones del resto de la liga.
Temporadas 2015-2025 · 11 campeones · solo consola, sin visuales.

Metodologia:
  - Para cada metrica se calcula el percentil de cada campeon vs su liga ese año.
  - Se reporta: percentil minimo (el "peor" campeon), mediana, y brecha de exclusividad
    (% campeones sobre umbral vs % liga sobre ese mismo umbral).
  - Ranking final por percentil minimo (cuanto peor puede ser un campeon y aun asi ganar).
"""

import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE   = "pbp_cache"
SEASONS = list(range(2015, 2026))

# ══════════════════════════════════════════════════════════════════════════════
# 1. CAMPEONES
# ══════════════════════════════════════════════════════════════════════════════
sch = pd.read_parquet(os.path.join(CACHE, "schedules.parquet"))
sb  = sch[sch["game_type"] == "SB"].copy()
sb["champion"] = sb.apply(
    lambda r: r["home_team"] if r["home_score"] > r["away_score"] else r["away_team"], axis=1
)
champions = sb[sb["season"].isin(SEASONS)].set_index("season")["champion"].to_dict()
print("Campeones:", champions)

# ══════════════════════════════════════════════════════════════════════════════
# 2. CARGA DE TODAS LAS FUENTES
# ══════════════════════════════════════════════════════════════════════════════

# ── 2a. game_logs (off_epa, def_epa, 3rd downs, turnovers, sack_rate)
print("Cargando game_logs...")
gl = pd.read_parquet(os.path.join(CACHE, "game_logs_all.parquet"))
ts = gl[gl["season"].isin(SEASONS)].groupby(["season", "team"]).agg(
    off_epa        = ("off_epa",        "mean"),
    def_epa        = ("def_epa",        "mean"),
    rz_epa         = ("rz_epa",         "mean"),
    third_conv     = ("third_conv",     "mean"),
    def_third_conv = ("def_third_conv", "mean"),
    turnovers_off  = ("turnovers_off",  "sum"),
    turnovers_def  = ("turnovers_def",  "sum"),
    sack_rate_def  = ("sack_rate_def",  "mean"),
).reset_index()
ts["to_diff"] = ts["turnovers_def"] - ts["turnovers_off"]

# ── 2b. Schedules: wins, pts_scored, pts_allowed, pt_differential
print("Cargando schedules (wins, puntos)...")
reg_games = sch[(sch["season"].isin(SEASONS)) & (sch["game_type"] == "REG")].copy()

home = reg_games[["season", "home_team", "home_score", "away_score"]].copy()
home.columns = ["season", "team", "pf", "pa"]
away = reg_games[["season", "away_team", "away_score", "home_score"]].copy()
away.columns = ["season", "team", "pf", "pa"]
scores = pd.concat([home, away])
scores["won"] = (scores["pf"] > scores["pa"]).astype(int)

sc_agg = scores.groupby(["season", "team"]).agg(
    wins         = ("won", "sum"),
    pts_scored   = ("pf",  "sum"),
    pts_allowed  = ("pa",  "sum"),
    games_played = ("won", "count"),
).reset_index()
sc_agg["pt_differential"] = sc_agg["pts_scored"] - sc_agg["pts_allowed"]
sc_agg["pts_scored_rank"]  = sc_agg.groupby("season")["pts_scored"].rank(ascending=False).astype(int)
sc_agg["pts_allowed_rank"] = sc_agg.groupby("season")["pts_allowed"].rank(ascending=True).astype(int)

ts = ts.merge(sc_agg, on=["season", "team"], how="left")

# ── 2c. Player stats: ypa, td_int, pass_yards, sacks_allowed, def_sacks
print("Cargando player_stats...")
ps = pd.read_parquet(os.path.join(CACHE, "player_stats.parquet"))
qb_reg = ps[(ps["season"].isin(SEASONS)) & (ps["season_type"] == "REG") &
            (ps["position"] == "QB")].copy()

qb_team = qb_reg.groupby(["season", "recent_team"]).agg(
    pass_yards  = ("passing_yards",  "sum"),
    pass_att    = ("attempts",       "sum"),
    pass_tds    = ("passing_tds",    "sum"),
    pass_ints   = ("interceptions",  "sum"),
    sacks_taken = ("sacks",          "sum"),
).reset_index()
qb_team["ypa"]           = qb_team["pass_yards"] / qb_team["pass_att"].clip(lower=1)
qb_team["td_int_ratio"]  = qb_team["pass_tds"]   / qb_team["pass_ints"].clip(lower=0.5)
qb_team["sacks_allowed"] = qb_team["sacks_taken"]
qb_team.rename(columns={"recent_team": "team"}, inplace=True)
ts = ts.merge(qb_team[["season", "team", "ypa", "td_int_ratio", "pass_yards", "sacks_allowed"]],
              on=["season", "team"], how="left")

# def_sacks: sacks que tomaron QBs rivales (= sacks generados por la defensa)
qb_opp = ps[(ps["season"].isin(SEASONS)) & (ps["season_type"] == "REG") &
            (ps["position"] == "QB") & ps["opponent_team"].notna()]
def_sacks_df = qb_opp.groupby(["season", "opponent_team"])["sacks"].sum().reset_index()
def_sacks_df.columns = ["season", "team", "def_sacks"]
ts = ts.merge(def_sacks_df, on=["season", "team"], how="left")

# ── 2d. Fullmetrics: yards_play_def, rz_td_pct
print("Cargando fullmetrics...")
fm_dfs = []
for yr in SEASONS:
    f = os.path.join(CACHE, f"fullmetrics_{yr}.parquet")
    if os.path.exists(f):
        fm_dfs.append(pd.read_parquet(f))
if fm_dfs:
    fm = pd.concat(fm_dfs, ignore_index=True)
    ts = ts.merge(fm[["season", "team", "yards_play_def", "rz_td_pct"]], on=["season", "team"], how="left")
else:
    ts["yards_play_def"] = np.nan
    ts["rz_td_pct"]      = np.nan

# ── 2e. Presión QB (FTN / pbp_part)
print("Cargando presion QB (pbp_part)...")
sch_slim = sch[["game_id", "home_team", "away_team", "game_type"]].copy()
part_dfs = []
for yr in SEASONS:
    f = os.path.join(CACHE, f"pbp_part_{yr}.parquet")
    if not os.path.exists(f):
        continue
    p = pd.read_parquet(f)[["nflverse_game_id", "possession_team", "was_pressure"]].copy()
    p["season"] = yr
    part_dfs.append(p)

if part_dfs:
    part_all = pd.concat(part_dfs, ignore_index=True)
    part_all = part_all.merge(sch_slim, left_on="nflverse_game_id", right_on="game_id", how="left")
    part_all = part_all[part_all["game_type"] == "REG"].copy()
    part_all["defteam"] = part_all.apply(
        lambda r: r["away_team"] if r["possession_team"] == r["home_team"] else r["home_team"], axis=1
    )
    part_all = part_all.dropna(subset=["was_pressure", "defteam"])
    press = part_all.groupby(["season", "defteam"]).agg(
        pressures = ("was_pressure", "sum"),
        dropbacks = ("was_pressure", "count"),
    ).reset_index()
    press["pressure_rate_def"] = press["pressures"] / press["dropbacks"].clip(lower=1)
    press.rename(columns={"defteam": "team"}, inplace=True)
    ts = ts.merge(press[["season", "team", "pressure_rate_def"]], on=["season", "team"], how="left")
else:
    ts["pressure_rate_def"] = np.nan

# ── 2f. Nuevas metricas (newmetrics cache)
print("Cargando newmetrics...")
nm_dfs = []
for yr in SEASONS:
    f = os.path.join(CACHE, f"newmetrics_{yr}.parquet")
    if os.path.exists(f):
        nm_dfs.append(pd.read_parquet(f))
if nm_dfs:
    nm_all = pd.concat(nm_dfs, ignore_index=True)
    gpt = (reg_games.melt(id_vars=["season"], value_vars=["home_team", "away_team"], value_name="team")
           .groupby(["season", "team"]).size().reset_index(name="n_games"))
    nm_all = nm_all.merge(gpt, on=["season", "team"], how="left")
    nm_all["qb_hits_pg"] = nm_all["qb_hits_def"] / nm_all["n_games"].clip(lower=1)
    nm_all["tfl_pg"]     = nm_all["tfl_def"]      / nm_all["n_games"].clip(lower=1)
    nm_all["fumbles_pg"] = nm_all["fumbles_lost"]  / nm_all["n_games"].clip(lower=1)
    nm_cols = ["season", "team", "success_rate_off", "success_rate_def",
               "qb_hits_pg", "tfl_pg", "fumbles_pg",
               "yac_avg", "air_yards_avg", "cpoe_avg",
               "drive_score_rate", "fourth_down_conv", "pen_yards_pg"]
    nm_cols = [c for c in nm_cols if c in nm_all.columns]
    ts = ts.merge(nm_all[nm_cols], on=["season", "team"], how="left")

# ── 2g. EPA por tipo de jugada (epa_type cache)
print("Cargando epa_type...")
et_dfs = []
for yr in SEASONS:
    f = os.path.join(CACHE, f"epa_type_{yr}.parquet")
    if os.path.exists(f):
        et_dfs.append(pd.read_parquet(f))
if et_dfs:
    et = pd.concat(et_dfs, ignore_index=True)
    ts = ts.merge(et[["season", "team", "def_pass_epa", "def_rush_epa"]], on=["season", "team"], how="left")
else:
    ts["def_pass_epa"] = np.nan
    ts["def_rush_epa"] = np.nan

# ── 2h. Metricas situacionales (situational cache)
print("Cargando situational...")
sit_dfs = []
for yr in SEASONS:
    f = os.path.join(CACHE, f"situational_{yr}.parquet")
    if os.path.exists(f):
        sit_dfs.append(pd.read_parquet(f))
if sit_dfs:
    sit = pd.concat(sit_dfs, ignore_index=True)
    sit_cols = ["season", "team", "third_conv_pct", "trailing_success_rate",
                "two_min_success_rate", "explosive_run_pct"]
    sit_cols = [c for c in sit_cols if c in sit.columns]
    ts = ts.merge(sit[sit_cols], on=["season", "team"], how="left")

# ── 2i. Jugadas explosivas (explosive cache)
print("Cargando explosive...")
exp_dfs = []
for yr in SEASONS:
    f = os.path.join(CACHE, f"explosive_{yr}.parquet")
    if os.path.exists(f):
        exp_dfs.append(pd.read_parquet(f))
if exp_dfs:
    exp = pd.concat(exp_dfs, ignore_index=True)
    exp_cols = ["season", "team", "exp_pass_off", "exp_run_off",
                "exp_pass_def", "exp_run_def"]
    exp_cols = [c for c in exp_cols if c in exp.columns]
    ts = ts.merge(exp[exp_cols], on=["season", "team"], how="left")

# ── 2j. Red Zone defensa (rz_def cache)
print("Cargando rz_def...")
rzd_dfs = []
for yr in SEASONS:
    f = os.path.join(CACHE, f"rz_def_{yr}.parquet")
    if os.path.exists(f):
        rzd_dfs.append(pd.read_parquet(f))
if rzd_dfs:
    rzd = pd.concat(rzd_dfs, ignore_index=True)
    ts = ts.merge(rzd[["season", "team", "rz_td_pct_def"]], on=["season", "team"], how="left")
else:
    ts["rz_td_pct_def"] = np.nan

ts["is_champ"] = ts.apply(lambda r: champions.get(r["season"]) == r["team"], axis=1)

print(f"\nDataset listo: {len(ts)} filas, {ts['is_champ'].sum()} campeones, {len(ts.columns)} columnas\n")

# ══════════════════════════════════════════════════════════════════════════════
# 3. DEFINICION DE METRICAS A ANALIZAR
# ══════════════════════════════════════════════════════════════════════════════
# (col, descripcion, lower_better)
METRICS = [
    # Eficiencia EPA
    ("off_epa",             "EPA ofensivo/jugada",          False),
    ("def_epa",             "EPA defensivo/jugada",         True),
    ("def_pass_epa",        "EPA DEF en pase",              True),
    ("def_rush_epa",        "EPA DEF en carrera",           True),
    ("rz_epa",              "EPA en Red Zone",              False),
    # Puntuacion
    ("pts_scored",          "Puntos anotados",              False),
    ("pts_allowed",         "Puntos cedidos",               True),
    ("pt_differential",     "Diferencial de puntos",        False),
    ("wins",                "Victorias",                    False),
    # Pase ofensivo
    ("ypa",                 "Yardas por intento",           False),
    ("td_int_ratio",        "Ratio TD/INT",                 False),
    ("pass_yards",          "Yardas totales de pase",       False),
    ("cpoe_avg",            "CPOE promedio",                False),
    ("air_yards_avg",       "Air yards promedio",           False),
    ("yac_avg",             "YAC promedio",                 False),
    # Defensa presion
    ("yards_play_def",      "Yardas/jugada cedidas",        True),
    ("pressure_rate_def",   "Tasa presion QB (FTN)",        False),
    ("def_sacks",           "Sacks defensivos",             False),
    ("sack_rate_def",       "Sack rate DEF",                False),
    ("qb_hits_pg",          "QB hits DEF/partido",          False),
    ("tfl_pg",              "Tackles for loss DEF/PJ",      False),
    # Red Zone
    ("rz_td_pct",           "RZ TD% ofensiva",              False),
    ("rz_td_pct_def",       "RZ TD% cedida DEF",            True),
    # Turnovers / balon
    ("turnovers_def",       "Turnovers forzados",           False),
    ("to_diff",             "Diferencial turnovers",        False),
    ("fumbles_pg",          "Fumbles perdidos/PJ",          True),
    # 3a bajada
    ("third_conv",          "3a bajada ofensiva %",         False),
    ("def_third_conv",      "3a bajada cedida DEF %",       True),
    # Proteccion OL
    ("sacks_allowed",       "Sacks permitidos",             True),
    # Explosivas
    ("exp_pass_off",        "Pases explosivos OFF (20+yd)", False),
    ("exp_run_off",         "Carreras explosivas OFF (10+)",False),
    ("exp_pass_def",        "Pases explosivos cedidos DEF", True),
    ("exp_run_def",         "Carreras explosivas cedidas",  True),
    # Situacional
    ("trailing_success_rate","Exito cuando van perdiendo",  False),
    ("two_min_success_rate", "Two-minute drill exito",      False),
    ("drive_score_rate",    "% drives con anotacion",       False),
    ("fourth_down_conv",    "Conversion en 4a bajada",      False),
    # Discipline
    ("pen_yards_pg",        "Penalidades yardas/PJ",        True),
    # Success rate
    ("success_rate_off",    "Success rate ofensivo",        False),
    ("success_rate_def",    "Success rate cedido DEF",      True),
]

# ══════════════════════════════════════════════════════════════════════════════
# 4. ANALISIS DISCRIMINANTE
# ══════════════════════════════════════════════════════════════════════════════
def pct_rank(series, val, lower_better=False):
    arr = series.dropna().values
    if len(arr) == 0:
        return np.nan
    p = (np.sum(arr < val) / len(arr)) * 100
    return round(100 - p if lower_better else p, 1)


champs_df = ts[ts["is_champ"]].sort_values("season").reset_index(drop=True)

results = []
for col, label, lb in METRICS:
    if col not in ts.columns:
        continue

    pcts        = []
    champ_vals  = []

    for _, row in champs_df.iterrows():
        val = row.get(col, np.nan)
        if pd.isna(val):
            continue
        sd = ts[ts["season"] == row["season"]]
        p  = pct_rank(sd[col], val, lb)
        if not np.isnan(p):
            pcts.append(p)
            champ_vals.append(val)

    if len(pcts) < 5:   # necesitamos al menos 5 campeones con datos
        continue

    p_min     = min(pcts)
    p_med     = float(np.median(pcts))
    n_below50 = sum(p < 50 for p in pcts)
    n_champs  = len(pcts)

    # Brecha de exclusividad: % campeones sobre umbral vs % liga
    umbral = max(champ_vals) if lb else min(champ_vals)
    n_meet_liga = []
    for yr in SEASONS:
        arr = ts[ts["season"] == yr][col].dropna().values
        if len(arr) == 0:
            continue
        if lb:
            n_meet_liga.append((arr <= umbral).sum() / len(arr) * 100)
        else:
            n_meet_liga.append((arr >= umbral).sum() / len(arr) * 100)
    pct_liga      = float(np.mean(n_meet_liga)) if n_meet_liga else np.nan
    pct_campeones = (sum(cv <= umbral if lb else cv >= umbral for cv in champ_vals) / n_champs * 100)
    brecha        = pct_campeones - pct_liga

    results.append({
        "col":          col,
        "label":        label,
        "lb":           lb,
        "p_min":        p_min,
        "p_med":        p_med,
        "n_below50":    n_below50,
        "n_champs":     n_champs,
        "umbral":       umbral,
        "pct_campeones":pct_campeones,
        "pct_liga":     pct_liga,
        "brecha":       brecha,
        "pcts":         pcts,
        "champ_vals":   champ_vals,
    })

# Ordenar por percentil minimo (descendente)
results.sort(key=lambda x: (x["p_min"], x["p_med"]), reverse=True)

# ══════════════════════════════════════════════════════════════════════════════
# 5. OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 105)
print(f"{'RANKING DE METRICAS POR PODER DISCRIMINANTE':^105}")
print(f"{'Temporadas 2015-2025 · 11 campeones del Super Bowl':^105}")
print("=" * 105)
print(f"{'Metrica':38s} {'N':>2}  {'p_min':>6}  {'p_med':>6}  {'Champ%':>7}  {'Liga%':>7}  {'Brecha':>7}  Fuerza")
print("-" * 105)

for r in results:
    if r["p_min"] >= 70:
        fuerza = "★★★ FUERTE"
    elif r["p_min"] >= 55:
        fuerza = "★★  MODERADO"
    elif r["p_min"] >= 40:
        fuerza = "★   DEBIL"
    else:
        fuerza = "    TRIVIAL"

    print(f"{r['label']:38s} {r['n_champs']:>2}  "
          f"{r['p_min']:>6.1f}  {r['p_med']:>6.1f}  "
          f"{r['pct_campeones']:>7.1f}%  {r['pct_liga']:>7.1f}%  "
          f"{r['brecha']:>+7.1f}  {fuerza}")

# ── Detalle por campeon de las metricas fuertes
fuertes = [r for r in results if r["p_min"] >= 55]

print()
print("=" * 105)
print(f"DETALLE POR CAMPEON — metricas con p_min >= 55 ({len(fuertes)} metricas)")
print("=" * 105)

for r in fuertes:
    col, label, lb = r["col"], r["label"], r["lb"]
    umbral_str = f"{r['umbral']:.3f}" if isinstance(r["umbral"], float) else str(r["umbral"])
    print(f"\n{label}  (p_min={r['p_min']:.0f}, p_med={r['p_med']:.0f}, "
          f"umbral={umbral_str}, liga cumple={r['pct_liga']:.0f}%, brecha={r['brecha']:+.0f}pp)")
    for _, row in champs_df.iterrows():
        val = row.get(col, np.nan)
        if pd.isna(val):
            continue
        sd = ts[ts["season"] == row["season"]]
        p  = pct_rank(sd[col], val, lb)
        cumple = "✓" if (val <= r["umbral"] if lb else val >= r["umbral"]) else "✗"
        print(f"  {row['season']} {row['team']:>3}: {val:>9.3f}  p{p:>5.1f}  {cumple}")

print()
print("=" * 105)
print("RESUMEN: METRICAS FUERTES (p_min >= 55)")
print("=" * 105)
for r in fuertes:
    print(f"  {r['label']:38s}  umbral: {r['umbral']:.3f}  "
          f"({r['pct_campeones']:.0f}% campeones vs {r['pct_liga']:.0f}% liga = {r['brecha']:+.0f}pp)")

print("\nFin del analisis.")
