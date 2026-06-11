"""
lab/generar_caches.py
Regenera los caches de pbp_cache que consumen discriminacion_total.py,
contenders.py y contenders_tracker.py:

  fullmetrics_{yr}  — yds/jugada DEF + Red Zone TD% ofensivo
  epa_type_{yr}     — EPA defensivo vs pase y vs carrera
  situational_{yr}  — 3er down, trailing, two-minute, carreras explosivas
  explosive_{yr}    — % jugadas explosivas OFF y DEF
  rz_def_{yr}       — Red Zone TD% defensivo
  newmetrics_{yr}   — penalidades, success rate, QB hits, fumbles, 4th down, YAC, CPOE...
  player_stats      — añade QB stats del año desde PBP si la temporada no existe

Consolida la lógica de los antiguos generar_caches_2025.py, nuevas_metricas.py
y formula_campeon.py (load_or_cache_full_pbp). Descarga el PBP completo del
año una sola vez y calcula todo de una pasada.

Uso (desde la raíz del proyecto):
  python lab/generar_caches.py 2026            # genera los caches que falten
  python lab/generar_caches.py 2026 --force    # regenera aunque existan
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
from pbp_loader import cargar_pbp

warnings.filterwarnings("ignore")

CACHE = "pbp_cache"

# Columnas del PBP necesarias para todos los caches
PBP_COLS = [
    "game_id", "drive", "season_type", "posteam", "defteam", "play_type",
    "epa", "yards_gained", "touchdown", "yardline_100",
    "score_differential", "down", "ydstogo", "first_down",
    "half_seconds_remaining", "qtr",
    "penalty", "penalty_team", "penalty_yards",
    "success", "fixed_drive", "drive_ended_with_score",
    "qb_hit", "sack", "fumble_lost",
    "fourth_down_converted", "fourth_down_failed",
    "yards_after_catch", "air_yards", "cpoe", "tackled_for_loss",
    "passer_player_id", "passer_player_name", "pass_touchdown", "interception",
]

NUM_COLS = [
    "epa", "yards_gained", "touchdown", "yardline_100", "score_differential",
    "down", "ydstogo", "first_down", "half_seconds_remaining", "qtr",
    "penalty", "penalty_yards", "success", "drive_ended_with_score",
    "qb_hit", "sack", "fumble_lost", "fourth_down_converted",
    "fourth_down_failed", "yards_after_catch", "air_yards", "cpoe",
    "tackled_for_loss", "pass_touchdown", "interception",
]


def descargar_pbp(yr: int) -> pd.DataFrame:
    raw, _ = cargar_pbp(yr, columns=[c for c in PBP_COLS], solo_reg=False)
    for c in NUM_COLS:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["touchdown"] = raw.get("touchdown", pd.Series(0, index=raw.index)).fillna(0)
    print(f"  Filas: {len(raw):,}")
    return raw


# ── 1. fullmetrics: yds/jugada DEF + RZ TD% ofensivo ─────────────────────────
def gen_fullmetrics(reg: pd.DataFrame, yr: int) -> pd.DataFrame:
    plays = reg[reg["play_type"].isin(["pass", "run"]) &
                reg["defteam"].notna() & reg["yards_gained"].notna()]
    ypd = plays.groupby("defteam").agg(
        total_yards_allowed=("yards_gained", "sum"),
        total_plays=("yards_gained", "count"),
    ).reset_index()
    ypd["yards_play_def"] = ypd["total_yards_allowed"] / ypd["total_plays"].clip(lower=1)
    ypd.rename(columns={"defteam": "team"}, inplace=True)

    # RZ TD% por trips: drive con >=1 jugada dentro de la 20 → ¿acabó en TD?
    rz_plays = reg[
        reg["play_type"].isin(["pass", "run"]) & reg["posteam"].notna() &
        reg["game_id"].notna() & reg["drive"].notna() & (reg["yardline_100"] <= 20)
    ].copy()
    rz_drives = rz_plays.groupby(["posteam", "game_id", "drive"]).size().reset_index(name="n")
    drive_tds = (reg[reg["touchdown"] == 1]
                 .groupby(["posteam", "game_id", "drive"])["touchdown"]
                 .sum().reset_index(name="drive_tds"))
    rz_drives = rz_drives.merge(drive_tds, on=["posteam", "game_id", "drive"], how="left")
    rz_drives["drive_tds"] = rz_drives["drive_tds"].fillna(0)
    rz_drives["scored_td"] = (rz_drives["drive_tds"] > 0).astype(int)
    rz_stat = rz_drives.groupby("posteam").agg(
        rz_trips=("scored_td", "count"), rz_tds=("scored_td", "sum")).reset_index()
    rz_stat["rz_td_pct"] = rz_stat["rz_tds"] / rz_stat["rz_trips"].clip(lower=1)
    rz_stat.rename(columns={"posteam": "team"}, inplace=True)

    fm = ypd.merge(rz_stat[["team", "rz_trips", "rz_tds", "rz_td_pct"]], on="team", how="outer")
    fm["season"] = yr
    return fm


# ── 2. epa_type: EPA defensivo vs pase y carrera ─────────────────────────────
def gen_epa_type(reg: pd.DataFrame, yr: int) -> pd.DataFrame:
    reg_epa = reg[reg["play_type"].isin(["pass", "run"]) &
                  reg["defteam"].notna() & reg["epa"].notna()]
    pass_def = reg_epa[reg_epa["play_type"] == "pass"].groupby("defteam")["epa"].mean().reset_index()
    pass_def.columns = ["team", "def_pass_epa"]
    rush_def = reg_epa[reg_epa["play_type"] == "run"].groupby("defteam")["epa"].mean().reset_index()
    rush_def.columns = ["team", "def_rush_epa"]
    et = pass_def.merge(rush_def, on="team", how="outer")
    et["season"] = yr
    return et


# ── 3. situational ────────────────────────────────────────────────────────────
def gen_situational(reg: pd.DataFrame, yr: int) -> pd.DataFrame:
    results = []
    for team in reg["posteam"].dropna().unique():
        td = reg[reg["posteam"] == team]
        d3 = td[td["down"] == 3]
        third_conv = d3["first_down"].sum() / len(d3) if len(d3) > 0 else np.nan
        trailing = td[(td["score_differential"] < 0) &
                      td["play_type"].isin(["pass", "run"]) & td["epa"].notna()]
        trailing_success = (trailing["epa"] > 0).mean() if len(trailing) > 0 else np.nan
        two_min = td[(td["half_seconds_remaining"] <= 120) & td["qtr"].isin([2, 4]) &
                     td["play_type"].isin(["pass", "run"]) & td["epa"].notna()]
        two_min_success = (two_min["epa"] > 0).mean() if len(two_min) > 0 else np.nan
        runs = td[td["play_type"] == "run"]
        explosive_run_pct = (runs["yards_gained"] >= 10).mean() if len(runs) > 0 else np.nan
        results.append({
            "team": team, "third_conv_pct": third_conv,
            "trailing_success_rate": trailing_success,
            "two_min_success_rate": two_min_success,
            "explosive_run_pct": explosive_run_pct,
            "n_3rd": len(d3), "n_trailing": len(trailing),
            "n_two_min": len(two_min), "n_runs": len(runs),
        })
    sit = pd.DataFrame(results)
    sit["season"] = yr
    return sit


# ── 4. explosive ──────────────────────────────────────────────────────────────
def gen_explosive(reg: pd.DataFrame, yr: int) -> pd.DataFrame:
    reg_e = reg[reg["play_type"].isin(["pass", "run"]) & reg["posteam"].notna()]
    exp_results = []
    for team in reg_e["posteam"].unique():
        td     = reg_e[reg_e["posteam"] == team]
        td_def = reg_e[reg_e["defteam"] == team]
        passes_off = td[td["play_type"] == "pass"]
        runs_off   = td[td["play_type"] == "run"]
        passes_def = td_def[td_def["play_type"] == "pass"]
        runs_def   = td_def[td_def["play_type"] == "run"]
        exp_results.append({
            "team": team,
            "exp_pass_off": (passes_off["yards_gained"] >= 20).mean() if len(passes_off) else np.nan,
            "exp_run_off":  (runs_off["yards_gained"]   >= 10).mean() if len(runs_off)   else np.nan,
            "exp_all_off":  (td["yards_gained"]         >= 15).mean() if len(td)         else np.nan,
            "exp_pass_def": (passes_def["yards_gained"] >= 20).mean() if len(passes_def) else np.nan,
            "exp_run_def":  (runs_def["yards_gained"]   >= 10).mean() if len(runs_def)   else np.nan,
        })
    exp = pd.DataFrame(exp_results)
    exp["season"] = yr
    return exp


# ── 5. rz_def: Red Zone TD% defensivo ────────────────────────────────────────
def gen_rz_def(reg: pd.DataFrame, yr: int) -> pd.DataFrame:
    rz_p = reg[reg["play_type"].isin(["pass", "run"]) &
               reg["defteam"].notna() & (reg["yardline_100"] <= 20)]
    rz_d = rz_p.groupby(["defteam", "game_id", "drive"]).size().reset_index(name="n")
    dtds = (reg[reg["touchdown"] == 1]
            .groupby(["posteam", "game_id", "drive"])["touchdown"]
            .sum().reset_index(name="drive_tds"))
    rz_d = rz_d.merge(dtds.rename(columns={"posteam": "off_team"}),
                      on=["game_id", "drive"], how="left")
    rz_d["scored_td"] = (rz_d["drive_tds"] > 0).fillna(False).astype(int)
    rz_def_df = rz_d.groupby("defteam").agg(
        rz_trips_def=("scored_td", "count"), rz_tds_allowed=("scored_td", "sum")).reset_index()
    rz_def_df["rz_td_pct_def"] = rz_def_df["rz_tds_allowed"] / rz_def_df["rz_trips_def"].clip(lower=1)
    rz_def_df.rename(columns={"defteam": "team"}, inplace=True)
    rz_def_df["season"] = yr
    return rz_def_df


# ── 6. newmetrics: penalidades, success rate, QB hits, YAC, 4th down... ───────
def gen_newmetrics(reg: pd.DataFrame, yr: int) -> pd.DataFrame:
    reg = reg[reg["posteam"].notna()].copy()
    for c in ["penalty_yards", "success", "qb_hit", "sack", "fumble_lost",
              "fourth_down_converted", "fourth_down_failed",
              "yards_after_catch", "air_yards", "cpoe",
              "tackled_for_loss", "drive_ended_with_score"]:
        if c in reg.columns:
            reg[c] = reg[c].fillna(0)

    results = []
    for team in reg["posteam"].dropna().unique():
        off  = reg[reg["posteam"] == team]
        def_ = reg[reg["defteam"] == team]
        plays_off = off[off["play_type"].isin(["pass", "run"])]
        plays_def = def_[def_["play_type"].isin(["pass", "run"])]

        if "penalty_team" in reg.columns:
            pen_count = (reg["penalty_team"] == team).sum()
            pen_yards_total = reg[reg["penalty_team"] == team]["penalty_yards"].sum()
        else:
            pen_count, pen_yards_total = np.nan, np.nan

        sr_off = plays_off["success"].mean() if len(plays_off) else np.nan
        sr_def = plays_def["success"].mean() if len(plays_def) else np.nan
        qb_hits_def = plays_def["qb_hit"].sum() if len(plays_def) else np.nan
        sacks_def   = plays_def["sack"].sum()   if len(plays_def) else np.nan
        fumbles_lost_off = off["fumble_lost"].sum()

        fd_attempts = off["fourth_down_converted"].sum() + off["fourth_down_failed"].sum()
        fd_conv_rate = off["fourth_down_converted"].sum() / fd_attempts if fd_attempts > 0 else np.nan

        passes_off = off[off["play_type"] == "pass"]
        yac_avg  = passes_off["yards_after_catch"].mean() if len(passes_off) else np.nan
        air_avg  = passes_off["air_yards"].mean()         if len(passes_off) else np.nan
        cpoe_avg = passes_off["cpoe"].mean()              if len(passes_off) else np.nan

        if "fixed_drive" in off.columns:
            drives = off.groupby("fixed_drive").first()
            drive_score_rate = drives["drive_ended_with_score"].mean()
        else:
            drive_score_rate = np.nan

        tfl_def = plays_def["tackled_for_loss"].sum() if len(plays_def) else np.nan

        results.append({
            "team": team,
            "pen_yards_committed": pen_yards_total,
            "pen_count": pen_count,
            "success_rate_off": sr_off,
            "success_rate_def": sr_def,
            "qb_hits_def": qb_hits_def,
            "sacks_def_raw": sacks_def,
            "fumbles_lost": fumbles_lost_off,
            "fourth_down_att": fd_attempts,
            "fourth_down_conv": fd_conv_rate,
            "yac_avg": yac_avg,
            "air_yards_avg": air_avg,
            "cpoe_avg": cpoe_avg,
            "drive_score_rate": drive_score_rate,
            "tfl_def": tfl_def,
        })
    nm = pd.DataFrame(results)
    nm["season"] = yr
    return nm


# ── 7. QB stats del año → player_stats (aprox. desde PBP) ────────────────────
def append_qb_stats(reg: pd.DataFrame, yr: int):
    ps_path = os.path.join(CACHE, "player_stats.parquet")
    ps_old = pd.read_parquet(ps_path)
    if yr in ps_old["season"].values:
        print(f"  player_stats ya tiene datos {yr} — sin cambios")
        return
    pp = reg[(reg["play_type"] == "pass") & reg["passer_player_id"].notna()].copy()
    pp["passing_yards"] = pp["yards_gained"].fillna(0)
    qb_agg = pp.groupby(["posteam", "passer_player_id"]).agg(
        passing_yards=("passing_yards", "sum"),
        attempts=("passer_player_id", "count"),
        passing_tds=("pass_touchdown", "sum"),
        interceptions=("interception", "sum"),
    ).reset_index()
    qb_agg.rename(columns={"posteam": "recent_team", "passer_player_id": "player_id"}, inplace=True)
    name_map = pp.groupby("passer_player_id")["passer_player_name"].agg(
        lambda s: s.value_counts().idxmax())
    qb_agg["player_name"]  = qb_agg["player_id"].map(name_map)
    qb_agg["season"]       = yr
    qb_agg["season_type"]  = "REG"
    qb_agg["position"]     = "QB"
    qb_agg["sacks"]        = 0.0  # aproximacion
    ps_new = pd.concat([ps_old, qb_agg], ignore_index=True)
    ps_new.to_parquet(ps_path, index=False)
    print(f"  player_stats actualizado con {len(qb_agg)} filas QB {yr}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
GENERATORS = {
    "fullmetrics": gen_fullmetrics,
    "epa_type":    gen_epa_type,
    "situational": gen_situational,
    "explosive":   gen_explosive,
    "rz_def":      gen_rz_def,
    "newmetrics":  gen_newmetrics,
}


def main():
    parser = argparse.ArgumentParser(description="Regenera los caches de la fórmula del campeón")
    parser.add_argument("year", type=int, help="Temporada a generar (ej: 2026)")
    parser.add_argument("--force", action="store_true", help="Regenerar aunque el cache exista")
    args = parser.parse_args()
    yr = args.year

    if not os.path.isdir(CACHE):
        sys.exit(f"ERROR: no se encuentra '{CACHE}/' — ejecutar desde la raíz del proyecto")

    pendientes = {name: os.path.join(CACHE, f"{name}_{yr}.parquet")
                  for name in GENERATORS}
    if not args.force:
        for name, path in list(pendientes.items()):
            if os.path.exists(path):
                print(f"{name}_{yr} ya existe — saltando (usa --force para regenerar)")
                del pendientes[name]

    ps_path = os.path.join(CACHE, "player_stats.parquet")
    necesita_qb = os.path.exists(ps_path) and \
        yr not in pd.read_parquet(ps_path, columns=["season"])["season"].values

    if not pendientes and not necesita_qb:
        print(f"Todos los caches de {yr} existen ya. Nada que hacer.")
        return

    raw = descargar_pbp(yr)
    reg = raw[raw["season_type"] == "REG"].copy()

    for name, path in pendientes.items():
        print(f"  Generando {name}_{yr}...")
        df = GENERATORS[name](reg, yr)
        df.to_parquet(path, index=False)
        print(f"    {name}_{yr} OK ({len(df)} equipos)")

    if necesita_qb:
        print(f"  Añadiendo QB stats {yr} a player_stats...")
        append_qb_stats(reg, yr)

    print(f"\nCaches {yr} listos.")


if __name__ == "__main__":
    main()
