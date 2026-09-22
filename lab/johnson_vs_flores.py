"""Ben Johnson (OC DET 2023-24, HC CHI 2025) contra Brian Flores (DC MIN 2023-).
Splits por dropback: blitz (FTN n_blitzers>0), rushers, cobertura (participation)."""
import sys; sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, numpy as np, pbp_loader as pl

JUEGOS = {2023: "DET", 2024: "DET", 2025: "CHI"}
MOFC = {"COVER_0", "COVER_1", "COVER_3", "COVER_9"}     # un solo safety profundo (0 = ninguno)
MOFO = {"COVER_2", "COVER_4", "COVER_6", "2_MAN"}

def temporada(s):
    df, _ = pl.cargar_pbp(s, avisar=False)
    db = df[(df.qb_dropback == 1) & df.epa.notna() & (df.play_type != "no_play")].copy()
    ftn, _ = pl.cargar_ftn(s)
    db = db.merge(ftn[["nflverse_game_id", "nflverse_play_id", "n_blitzers", "n_pass_rushers", "is_play_action"]],
                  left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"], how="left")
    part, _ = pl.cargar_participation(s)
    db = db.merge(part[["nflverse_game_id", "play_id", "defense_coverage_type", "number_of_pass_rushers"]],
                  left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"], how="left",
                  suffixes=("", "_p"))
    db["blitz"] = db.n_blitzers > 0
    db["cob"] = np.where(db.defense_coverage_type.isin(MOFC), "MOFC",
                np.where(db.defense_coverage_type.isin(MOFO), "MOFO", None))
    return db

def resumen(x, et):
    f = lambda m: (round(x.loc[m, "epa"].mean(), 3), int(m.sum()))
    return {"tramo": et, "db": len(x), "EPA/db": round(x.epa.mean(), 3),
            "blitz%": round(x.blitz.mean() * 100, 1), "blitz": f(x.blitz), "no_blitz": f(x.n_blitzers == 0),
            "4 rush": f(x.n_pass_rushers == 4), "5+ rush": f(x.n_pass_rushers >= 5), "<=3 rush": f(x.n_pass_rushers <= 3),
            "MOFC": f(x.cob == "MOFC"), "MOFO": f(x.cob == "MOFO"),
            "sack%": round(x.sack.mean() * 100, 1)}

todos = []
for s, off in JUEGOS.items():
    db = temporada(s)
    g = db[(db.posteam == off) & (db.defteam == "MIN")]
    for gid, x in g.groupby("game_id"):
        pts = x[["posteam_score_post", "defteam_score_post"]].iloc[-1].tolist()
        print(gid, "QB:", x.passer_player_name.value_counts().index[0], resumen(x, gid))
    todos.append(g)
    if s == 2025:
        liga = db
T = pd.concat(todos)
print("\nTOTAL 6 partidos:", resumen(T, "total"))
print("Solo Goff (2023-24):", resumen(T[T.season < 2025], "DET"))
print("Solo CHI 2025:", resumen(T[T.season == 2025], "CHI"))

# Flores: blitz% de MIN en 2025 vs liga; Caleb vs blitz 2025
d25 = liga
bl = d25.groupby("defteam").blitz.mean().sort_values(ascending=False) * 100
print("\nMIN blitz% 2025:", round(bl["MIN"], 1), "rank", list(bl.index).index("MIN") + 1, "liga media", round(bl.mean(), 1))
q = d25[d25.blitz].groupby("passer_player_name").agg(n=("epa", "size"), epa=("epa", "mean"), sack=("sack", "mean"))
q = q[q.n >= 150].sort_values("epa", ascending=False); q["rk"] = range(1, len(q) + 1)
print(q.head(6).round(3)); print("Caleb:", q.loc["C.Williams"].round(3).to_dict(), "de", len(q))
qs = q.sort_values("sack"); print("sack% blitz rank Caleb:", list(qs.index).index("C.Williams") + 1)
