# lab/manning_experimentos.py
# Experimentos para mejorar Manning Bot contra su linea base: --bench
# (65,7% vs 68,2% del mercado; 37,5% en discrepancias).
#
# Fase 1: entrena los 4 folds walk-forward UNA vez y guarda las probs por
#         partido en lab/manning_probs_wf.parquet. Todo lo demas se prueba
#         sobre ese fichero sin reentrenar.
# Fase 2: diagnostico (por semana, por confianza, calibracion) y variantes
#         baratas (mezcla bot+mercado, calibracion isotonica).

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression

import Manning_bot as MB

PROBS = "lab/manning_probs_wf.parquet"


def construir_matrix():
    schedules = MB.load_schedules()
    MB.resolver_temporadas(schedules)
    game_logs = pd.read_parquet(MB.CACHE_DIR / "game_logs_all.parquet")
    rolling       = MB.add_season_carryover(MB.compute_rolling(game_logs))
    sched_logs    = MB.build_schedule_logs(schedules)
    sched_rolling = MB.compute_schedule_rolling(sched_logs)
    elo_df        = MB.compute_elo(schedules)
    sos_df        = MB.compute_sos(elo_df)
    qb_rolling    = MB.load_qb_rolling()
    train_df = MB.build_features(schedules, rolling, sched_rolling,
                                 elo_df, qb_rolling, sos_df,
                                 seasons=MB.SEASONS_TRAIN)
    return train_df


def fase1_generar_probs():
    train_df = construir_matrix()
    X_all    = train_df[MB.FEATURE_COLS].fillna(0)
    y_all    = train_df["home_win"]
    seas_all = train_df["season"]

    partes = []
    for val_yr, train_yrs in MB.folds_walk_forward(MB.SEASONS_TRAIN, n=4):
        print(f"  Entrenando fold {val_yr}...")
        mask_tr = seas_all.isin(train_yrs)
        mask_vl = seas_all == val_yr
        modelo  = MB.train_model(X_all[mask_tr], y_all[mask_tr])
        p_bot   = modelo.predict_proba(X_all[mask_vl])[:, 1]
        sub = train_df.loc[mask_vl, ["game_id", "season", "week",
                                     "home_team", "away_team", "home_win"]].copy()
        sub["p_bot"] = p_bot
        sub["p_mkt"] = X_all.loc[mask_vl, "home_impl_prob"].to_numpy()
        partes.append(sub)
    df = pd.concat(partes, ignore_index=True)
    df.to_parquet(PROBS, index=False)
    print(f"Guardado {PROBS}: {len(df)} partidos")
    return df


def metricas(y, p, etiqueta):
    acc = ((p >= 0.5).astype(int) == y).mean()
    print(f"  {etiqueta:<34} Acc={acc*100:5.1f}%  Brier={brier_score_loss(y,p):.4f}"
          f"  LogLoss={log_loss(y,p):.4f}")
    return acc


def fase2(df):
    y  = df["home_win"].to_numpy()
    pb = df["p_bot"].to_numpy()
    pm = df["p_mkt"].to_numpy()

    print("\n== A. DIAGNOSTICO: donde pierde el bot ==")
    df = df.copy()
    df["ok_bot"] = ((pb >= 0.5).astype(int) == y)
    df["ok_mkt"] = ((pm >= 0.5).astype(int) == y)
    df["tramo"]  = pd.cut(df["week"], [0, 4, 17, 19],
                          labels=["sem 1-4", "sem 5-17", "sem 18"])
    print(df.groupby("tramo", observed=True)
            .agg(n=("ok_bot","size"),
                 acc_bot=("ok_bot","mean"), acc_mkt=("ok_mkt","mean"))
            .assign(acc_bot=lambda d:(d.acc_bot*100).round(1),
                    acc_mkt=lambda d:(d.acc_mkt*100).round(1)))

    disc = df[(pb >= 0.5) != (pm >= 0.5)]
    print(f"\n  Discrepancias por tramo (n={len(disc)}):")
    print(disc.groupby("tramo", observed=True)
              .agg(n=("ok_bot","size"), acc_bot_disc=("ok_bot","mean"))
              .assign(acc_bot_disc=lambda d:(d.acc_bot_disc*100).round(1)))

    # confianza del bot al discrepar: fuerte (>60%) o timida (50-60%)?
    conf = np.abs(disc["p_bot"] - 0.5)
    fuerte = disc[conf >= 0.10]
    timida = disc[conf < 0.10]
    print(f"\n  Discrepancia timida (bot 50-60%): n={len(timida)}, "
          f"acierta {timida['ok_bot'].mean()*100:.1f}%")
    print(f"  Discrepancia fuerte (bot >60%):   n={len(fuerte)}, "
          f"acierta {fuerte['ok_bot'].mean()*100:.1f}%" if len(fuerte)
          else "  Sin discrepancias fuertes")

    print("\n  Calibracion del bot por tramos de prob:")
    bins = pd.cut(pb, np.arange(0.2, 0.85, 0.1))
    cal = pd.DataFrame({"bin": bins, "y": y}).groupby("bin", observed=True) \
            .agg(n=("y","size"), real=("y","mean"))
    cal["real"] = (cal["real"]*100).round(1)
    print(cal)

    print("\n== B. MEZCLA bot + mercado (sin reentrenar) ==")
    metricas(y, pb, "bot solo (baseline)")
    metricas(y, pm, "mercado solo")
    for w in [0.1, 0.2, 0.3, 0.4, 0.5]:
        metricas(y, w*pb + (1-w)*pm, f"mezcla {int(w*100)}% bot / {int((1-w)*100)}% mkt")

    print("\n== C. CALIBRACION isotonica (2022-2024 -> 2025) ==")
    ent = df["season"] < 2025
    val = df["season"] == 2025
    iso = IsotonicRegression(out_of_bounds="clip").fit(pb[ent], y[ent])
    p_cal = np.clip(iso.predict(pb[val]), 0.01, 0.99)
    metricas(y[val], pb[val], "2025 bot sin calibrar")
    metricas(y[val], p_cal,   "2025 bot calibrado")
    metricas(y[val], pm[val], "2025 mercado")


if __name__ == "__main__":
    if os.path.exists(PROBS) and "--regen" not in sys.argv:
        print(f"Usando {PROBS} existente (pasa --regen para reentrenar folds)")
        df = pd.read_parquet(PROBS)
    else:
        df = fase1_generar_probs()
    fase2(df)
