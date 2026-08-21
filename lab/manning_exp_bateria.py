# lab/manning_exp_bateria.py
# Bateria de experimentos para cerrar el hueco bot-mercado (65,5% vs 68,0%).
# Todas las variantes se evaluan walk-forward 2022-2025 contra las probs
# baseline de lab/manning_probs_wf.parquet. Ninguna toca produccion.
#
#  B  margen: regresion sobre el resultado (XGB+Ridge) -> prob via normal
#  C  features v2: success rate, EPA neutral (wp 5-95%), EPA downs 1-2,
#     equipos especiales — todo desde pbp_full local
#  D  B + C combinadas
#  E  poda: fuera las 8 features con menos importancia (medida en el fold)
#  F  promedio de probs C y D

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss

import Manning_bot as MB

PROBS_BASE = "lab/manning_probs_wf.parquet"
LOGS_V2    = "lab/game_logs_v2.parquet"

EXTRA = ["succ_off", "succ_def", "neutral_off_epa", "neutral_def_epa",
         "early_pass_off", "early_pass_def", "st_epa"]


def construir_logs_v2():
    """Metricas de estabilidad por team-week desde los pbp_full locales."""
    if os.path.exists(LOGS_V2):
        return pd.read_parquet(LOGS_V2)
    filas = []
    for s in range(2015, 2026):
        print(f"  logs v2 {s}...")
        d = pd.read_parquet(f"pbp_cache/pbp_full_{s}.parquet",
                            columns=["season", "week", "season_type", "posteam",
                                     "defteam", "play_type", "epa", "success",
                                     "wp", "down", "special_teams_play"])
        d = d[d["season_type"] == "REG"]
        d["posteam"] = d["posteam"].replace(MB.TEAM_MAP)
        d["defteam"] = d["defteam"].replace(MB.TEAM_MAP)
        pr = d[d["play_type"].isin(["pass", "run"])]
        neutral = pr[(pr["wp"] >= 0.05) & (pr["wp"] <= 0.95)]
        early_p = pr[(pr["play_type"] == "pass") & (pr["down"].isin([1, 2]))]
        st = d[d["special_teams_play"] == 1]

        def agg(df, key, col):
            return df.groupby(["season", "week", key])[col].mean()

        off = pd.DataFrame({
            "succ_off":        agg(pr, "posteam", "success"),
            "neutral_off_epa": agg(neutral, "posteam", "epa"),
            "early_pass_off":  agg(early_p, "posteam", "epa"),
            "st_epa":          agg(st, "posteam", "epa"),
        })
        deff = pd.DataFrame({
            "succ_def":        agg(pr, "defteam", "success"),
            "neutral_def_epa": agg(neutral, "defteam", "epa"),
            "early_pass_def":  agg(early_p, "defteam", "epa"),
        })
        deff.index.names = off.index.names
        filas.append(off.join(deff, how="outer").reset_index()
                        .rename(columns={"posteam": "team"}))
    v2 = pd.concat(filas, ignore_index=True)
    v2.to_parquet(LOGS_V2, index=False)
    return v2


def matrix(con_extra):
    schedules = MB.load_schedules()
    MB.resolver_temporadas(schedules)
    logs = pd.read_parquet(MB.CACHE_DIR / "game_logs_all.parquet")
    if con_extra:
        logs = logs.merge(construir_logs_v2(), on=["season", "week", "team"], how="left")
        MB.PBP_STAT_COLS = MB.PBP_STAT_COLS + EXTRA   # rolling + carryover + h_/a_
    rolling       = MB.add_season_carryover(MB.compute_rolling(logs))
    sched_logs    = MB.build_schedule_logs(schedules)
    sched_rolling = MB.compute_schedule_rolling(sched_logs)
    elo_df        = MB.compute_elo(schedules)
    sos_df        = MB.compute_sos(elo_df)
    qb_rolling    = MB.load_qb_rolling()
    df = MB.build_features(schedules, rolling, sched_rolling,
                           elo_df, qb_rolling, sos_df, seasons=MB.SEASONS_TRAIN)
    feats = list(MB.FEATURE_COLS)
    if con_extra:
        for c in EXTRA:
            df[f"d_{c}"] = df[f"h_{c}"].fillna(0) - df[f"a_{c}"].fillna(0)
            feats.append(f"d_{c}")
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    return df, feats


def entrenar_margen(Xtr, mtr):
    reg = xgb.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                           gamma=1.0, reg_alpha=0.1, reg_lambda=2.0,
                           random_state=42, n_jobs=-1, verbosity=0)
    rid = Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=10.0))])
    reg.fit(Xtr, mtr)
    rid.fit(Xtr, mtr)
    return reg, rid


def probs_margen(modelos, Xtr, mtr, Xvl):
    reg, rid = modelos
    pred_tr = (reg.predict(Xtr) + rid.predict(Xtr)) / 2
    sigma   = float(np.std(mtr - pred_tr, ddof=1))
    pred_vl = (reg.predict(Xvl) + rid.predict(Xvl)) / 2
    return norm.cdf(pred_vl / sigma)


def walk_forward(df, feats, modo, podar=0):
    """modo: 'clf' o 'margen'. podar: quita las N menos importantes (medidas
    con un XGB rapido SOLO sobre el train del fold — sin mirar validacion)."""
    X_all = df[feats].fillna(0)
    y_all = df["home_win"]
    m_all = df["result"]
    seas  = df["season"]
    out = []
    for val_yr, train_yrs in MB.folds_walk_forward(MB.SEASONS_TRAIN, n=4):
        tr = seas.isin(train_yrs)
        vl = seas == val_yr
        cols = feats
        if podar:
            probe = xgb.XGBClassifier(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, random_state=42,
                                      n_jobs=-1, verbosity=0)
            probe.fit(X_all[tr], y_all[tr])
            imp = pd.Series(probe.feature_importances_, index=feats)
            cols = imp.sort_values(ascending=False).index[:-podar].tolist()
        Xtr, Xvl = X_all.loc[tr, cols], X_all.loc[vl, cols]
        if modo == "clf":
            p = MB.train_model(Xtr, y_all[tr]).predict_proba(Xvl)[:, 1]
        else:
            p = probs_margen(entrenar_margen(Xtr, m_all[tr]), Xtr, m_all[tr], Xvl)
        sub = df.loc[vl, ["game_id", "season", "week", "home_win"]].copy()
        sub["p"] = p
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def informe(nombre, res, base):
    m = res.merge(base[["game_id", "p_bot", "p_mkt"]], on="game_id")
    y = m["home_win"].to_numpy()
    acc  = ((m["p"] >= 0.5) == y).mean()
    br   = brier_score_loss(y, m["p"])
    mid  = m[(m["week"] >= 5) & (m["week"] <= 17)]
    accm = ((mid["p"] >= 0.5) == mid["home_win"]).mean()
    disc = m[(m["p"] >= 0.5) != (m["p_mkt"] >= 0.5)]
    accd = ((disc["p"] >= 0.5) == disc["home_win"]).mean() if len(disc) else float("nan")
    print(f"  {nombre:<26} Acc={acc*100:5.1f}%  Brier={br:.4f}  "
          f"sem5-17={accm*100:5.1f}%  disc n={len(disc):>3} acc={accd*100:5.1f}%")
    return m


if __name__ == "__main__":
    base = pd.read_parquet(PROBS_BASE)

    print("\n== Matrix con features v2 ==")
    df2, feats2 = matrix(con_extra=True)
    feats_base  = list(MB.FEATURE_COLS)
    print(f"  {len(df2)} partidos, {len(feats2)} features (base {len(feats_base)})")

    y_ref = base["home_win"].to_numpy()
    mid   = base[(base["week"] >= 5) & (base["week"] <= 17)]
    print()
    for etq, col in [("A baseline (clf, 30 feats)", "p_bot"), ("mercado", "p_mkt")]:
        acc  = ((base[col] >= 0.5) == base["home_win"]).mean()
        accm = ((mid[col] >= 0.5) == mid["home_win"]).mean()
        print(f"  {etq:<26} Acc={acc*100:5.1f}%  "
              f"Brier={brier_score_loss(y_ref, base[col]):.4f}  sem5-17={accm*100:5.1f}%")

    print("\n== Variantes ==")
    rB = walk_forward(df2, feats_base, "margen")
    mB = informe("B margen (feats base)", rB, base)
    rC = walk_forward(df2, feats2, "clf")
    mC = informe("C clf + features v2", rC, base)
    rD = walk_forward(df2, feats2, "margen")
    mD = informe("D margen + features v2", rD, base)
    rE = walk_forward(df2, feats_base, "clf", podar=8)
    mE = informe("E clf podado (-8 feats)", rE, base)

    rF = rC.merge(rD[["game_id", "p"]], on="game_id", suffixes=("", "_d"))
    rF["p"] = (rF["p"] + rF["p_d"]) / 2
    informe("F promedio C+D", rF[["game_id", "season", "week", "home_win", "p"]], base)

    mC.to_parquet("lab/manning_probs_wf_C.parquet", index=False)
    mD.to_parquet("lab/manning_probs_wf_D.parquet", index=False)
    print("\nProbs de C y D guardadas en lab/ para iterar.")
