# lab/manning_exp_qb_out.py
# Experimento: feature d_qb_out — el QB titular esperado esta Out/Doubtful en
# el parte de lesiones de esta semana. Es el unico canal de informacion pregame
# que el mercado tiene y el bot no: todas sus features son rolling del pasado.
# Se mide contra las probs walk-forward guardadas por manning_experimentos.py.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

import Manning_bot as MB

INJ_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
           "injuries/injuries_{s}.parquet")
PROBS_BASE = "lab/manning_probs_wf.parquet"
PROBS_EXP  = "lab/manning_probs_wf_qbout.parquet"


def cargar_lesiones_qb(seasons):
    """(season, week, team, gsis_id) de QBs Out/Doubtful. Cachea en pbp_cache/."""
    partes = []
    for s in seasons:
        cache = MB.CACHE_DIR / f"injuries_{s}.parquet"
        try:
            if cache.exists():
                d = pd.read_parquet(cache)
            else:
                d = pd.read_parquet(INJ_URL.format(s=s))
                d.to_parquet(cache, index=False)
        except Exception as e:
            print(f"  injuries_{s}: NO disponible ({type(e).__name__}) — feature=0 ese año")
            continue
        d = d[(d["position"] == "QB") &
              (d["report_status"].isin(["Out", "Doubtful"]))]
        partes.append(d[["season", "week", "team", "gsis_id"]])
    inj = pd.concat(partes, ignore_index=True)
    inj["team"] = inj["team"].replace(MB.TEAM_MAP)
    print(f"  Lesiones QB Out/Doubtful: {len(inj)} filas, "
          f"temporadas {sorted(inj.season.unique())}")
    return inj


def titular_esperado():
    """Para cada (season, team, week): QB lider de intentos del ULTIMO partido
    jugado antes de esa semana (semana 1 hereda el cierre del año anterior)."""
    p = pd.read_parquet(MB.CACHE_DIR / "player_stats.parquet",
                        columns=["player_id", "position", "recent_team",
                                 "season", "week", "season_type", "attempts"])
    q = p[(p["position"] == "QB") & (p["season_type"] == "REG") &
          (p["season"] >= 2014) & (p["attempts"] > 0)].copy()
    q["recent_team"] = q["recent_team"].replace(MB.TEAM_MAP)
    # lider de intentos de cada team-week jugado
    lid = (q.sort_values("attempts", ascending=False)
             .drop_duplicates(["season", "week", "recent_team"])
             [["season", "week", "recent_team", "player_id"]]
             .rename(columns={"recent_team": "team", "player_id": "qb_titular"}))
    # orden cronologico global y forward-fill por equipo hacia la semana SIGUIENTE
    lid["orden"] = lid["season"] * 100 + lid["week"]
    filas = []
    for team, g in lid.groupby("team"):
        g = g.sort_values("orden")
        # el titular ESPERADO en la semana w es el lider del partido anterior
        g["qb_esperado"] = g["qb_titular"].shift(1)
        filas.append(g)
    return pd.concat(filas, ignore_index=True)[
        ["season", "week", "team", "qb_esperado"]]


def construir_qb_out(seasons):
    inj = cargar_lesiones_qb(seasons)
    tit = titular_esperado()
    m = tit.merge(inj, left_on=["season", "week", "team", "qb_esperado"],
                  right_on=["season", "week", "team", "gsis_id"], how="left")
    m["qb_out"] = m["gsis_id"].notna().astype(int)
    print(f"  Team-weeks con titular Out/Doubtful: {m['qb_out'].sum()} "
          f"de {len(m)} ({m['qb_out'].mean()*100:.1f}%)")
    return m[["season", "week", "team", "qb_out"]]


def main():
    schedules = MB.load_schedules()
    MB.resolver_temporadas(schedules)
    qb_out = construir_qb_out(MB.SEASONS_TRAIN)

    game_logs = pd.read_parquet(MB.CACHE_DIR / "game_logs_all.parquet")
    rolling       = MB.add_season_carryover(MB.compute_rolling(game_logs))
    sched_logs    = MB.build_schedule_logs(schedules)
    sched_rolling = MB.compute_schedule_rolling(sched_logs)
    elo_df        = MB.compute_elo(schedules)
    sos_df        = MB.compute_sos(elo_df)
    qb_rolling    = MB.load_qb_rolling()
    df = MB.build_features(schedules, rolling, sched_rolling,
                           elo_df, qb_rolling, sos_df, seasons=MB.SEASONS_TRAIN)

    df = df.merge(qb_out.rename(columns={"team": "home_team", "qb_out": "h_qb_out"}),
                  on=["season", "week", "home_team"], how="left")
    df = df.merge(qb_out.rename(columns={"team": "away_team", "qb_out": "a_qb_out"}),
                  on=["season", "week", "away_team"], how="left")
    df["h_qb_out"] = df["h_qb_out"].fillna(0)
    df["a_qb_out"] = df["a_qb_out"].fillna(0)
    df["d_qb_out"] = df["h_qb_out"] - df["a_qb_out"]
    n_flag = (df["d_qb_out"] != 0).sum()
    # sanity: cuanto pierde de verdad un equipo sin su titular?
    con = df[df["d_qb_out"] == 1]; sin = df[df["d_qb_out"] == -1]
    wr = (con["home_win"].sum() + (1 - sin["home_win"]).sum()) / (len(con) + len(sin))
    print(f"  Partidos con exactamente un titular fuera: {n_flag} "
          f"— ese equipo gana el {wr*100:.1f}%")

    FEATS = MB.FEATURE_COLS + ["d_qb_out"]
    X_all    = df[FEATS].fillna(0)
    y_all    = df["home_win"]
    seas_all = df["season"]

    partes = []
    for val_yr, train_yrs in MB.folds_walk_forward(MB.SEASONS_TRAIN, n=4):
        print(f"  Entrenando fold {val_yr} (con d_qb_out)...")
        mask_tr = seas_all.isin(train_yrs)
        mask_vl = seas_all == val_yr
        modelo  = MB.train_model(X_all[mask_tr], y_all[mask_tr])
        sub = df.loc[mask_vl, ["game_id", "season", "week", "home_win", "d_qb_out"]].copy()
        sub["p_exp"] = modelo.predict_proba(X_all[mask_vl])[:, 1]
        sub["p_mkt"] = X_all.loc[mask_vl, "home_impl_prob"].to_numpy()
        partes.append(sub)
    exp = pd.concat(partes, ignore_index=True)
    exp.to_parquet(PROBS_EXP, index=False)

    base = pd.read_parquet(PROBS_BASE)[["game_id", "p_bot"]]
    m = exp.merge(base, on="game_id")
    y = m["home_win"].to_numpy()

    def linea(etq, p):
        acc = ((p >= 0.5).astype(int) == y).mean()
        print(f"  {etq:<28} Acc={acc*100:5.1f}%  Brier={brier_score_loss(y, p):.4f}")

    print(f"\n== RESULTADO ({len(m)} partidos walk-forward 2022-2025) ==")
    linea("baseline (sin d_qb_out)", m["p_bot"].to_numpy())
    linea("experimento (con d_qb_out)", m["p_exp"].to_numpy())
    linea("mercado", m["p_mkt"].to_numpy())

    sub = m[m["d_qb_out"] != 0]
    if len(sub):
        ys = sub["home_win"].to_numpy()
        print(f"\n  Solo partidos con titular fuera (n={len(sub)}):")
        for etq, col in [("baseline", "p_bot"), ("experimento", "p_exp"), ("mercado", "p_mkt")]:
            acc = ((sub[col] >= 0.5).astype(int) == ys).mean()
            print(f"    {etq:<12} Acc={acc*100:5.1f}%  Brier={brier_score_loss(ys, sub[col]):.4f}")


if __name__ == "__main__":
    main()
