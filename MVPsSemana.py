# MVPsSemana.py
# Líderes semanales de EPA — Ataque, Defensa y Equipos Especiales
# Ataque: 50% QB / 50% receptor en pases, 100% portador en carreras
# Defensa: EPA negativo generado por intercepciones y sacks
# Equipos especiales: retornos, FGs, XPs y punts

import pandas as pd
import numpy as np

# === Config ===
SEASON = 2025
URL    = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"

def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def pick_col(df, *cands):
    for c in cands:
        if c and c in df.columns:
            return c
    return None

def make_key(name_col, team_col):
    """Combina nombre y equipo en 'Nombre (TEAM)'."""
    return name_col.str.strip() + " (" + team_col.fillna("?") + ")"


def calc_ataque(d, play_type, passer, receiver, rusher, posteam):
    credits = {}

    # ── Pases: QB 50% + receptor 50% (o QB 100% si no hay receptor) ──
    if passer and play_type:
        passes = d[(d[play_type] == "pass") & d["epa"].notna() & d[passer].notna()].copy()
        if not passes.empty:
            passes["_key_qb"] = make_key(passes[passer], passes[posteam] if posteam else pd.Series("", index=passes.index))

            # QB siempre 50%
            qb_base = passes.groupby("_key_qb")["epa"].sum() * 0.5

            # QB extra 50% cuando no hay receptor
            no_rec = passes[passes[receiver].isna()] if receiver else passes
            qb_extra = no_rec.groupby("_key_qb")["epa"].sum() * 0.5

            qb_total = qb_base.add(qb_extra, fill_value=0)
            for k, v in qb_total.items():
                credits[k] = credits.get(k, 0.0) + v

            # Receptor 50%
            if receiver:
                rec_plays = passes[passes[receiver].notna()].copy()
                if not rec_plays.empty:
                    rec_plays["_key_rec"] = make_key(rec_plays[receiver], rec_plays[posteam] if posteam else pd.Series("", index=rec_plays.index))
                    rec_total = rec_plays.groupby("_key_rec")["epa"].sum() * 0.5
                    for k, v in rec_total.items():
                        credits[k] = credits.get(k, 0.0) + v

    # ── Carreras: 100% al portador ──
    if rusher and play_type:
        runs = d[(d[play_type] == "run") & d["epa"].notna() & d[rusher].notna()].copy()
        if not runs.empty:
            runs["_key"] = make_key(runs[rusher], runs[posteam] if posteam else pd.Series("", index=runs.index))
            run_total = runs.groupby("_key")["epa"].sum()
            for k, v in run_total.items():
                credits[k] = credits.get(k, 0.0) + v

    return credits


def calc_defensa(d, interception, sack_p, defteam):
    credits = {}

    # Intercepciones: EPA negativo
    if interception:
        ints = d[d[interception].notna() & d["epa"].notna()].copy()
        if not ints.empty:
            ints["_key"] = make_key(ints[interception], ints[defteam] if defteam else pd.Series("", index=ints.index))
            int_total = ints.groupby("_key")["epa"].sum() * -1
            for k, v in int_total.items():
                credits[k] = credits.get(k, 0.0) + v

    # Sacks: EPA negativo
    if sack_p and "sack" in d.columns:
        sacks = d[(d["sack"] == 1) & d["epa"].notna() & d[sack_p].notna()].copy()
        if not sacks.empty:
            sacks["_key"] = make_key(sacks[sack_p], sacks[defteam] if defteam else pd.Series("", index=sacks.index))
            sack_total = sacks.groupby("_key")["epa"].sum() * -1
            for k, v in sack_total.items():
                credits[k] = credits.get(k, 0.0) + v

    return credits


def calc_st(d, play_type, kr_ret, pr_ret, kicker, punter, posteam):
    credits = {}

    def _add(plays, name_col, team_col, multiplier=1.0):
        if name_col and not plays.empty:
            sub = plays[plays[name_col].notna()].copy()
            if not sub.empty:
                sub["_key"] = make_key(sub[name_col], sub[team_col] if team_col else pd.Series("", index=sub.index))
                totals = sub.groupby("_key")["epa"].sum() * multiplier
                for k, v in totals.items():
                    credits[k] = credits.get(k, 0.0) + v

    if play_type:
        base = d[d["epa"].notna()]
        _add(base[base[play_type] == "kickoff_return"],  kr_ret,  posteam)
        _add(base[base[play_type] == "punt_return"],     pr_ret,  posteam)
        _add(base[base[play_type] == "field_goal"],      kicker,  posteam)
        _add(base[base[play_type] == "extra_point"],     kicker,  posteam)
        _add(base[base[play_type] == "punt"],            punter,  posteam)

    return credits


def print_top(label, credits, n=3):
    if not credits:
        print(f"  {label}: sin datos")
        return
    series = pd.Series(credits).sort_values(ascending=False)
    leader = series.index[0]
    leader_val = series.iloc[0]
    print(f"  {label}: {leader}  EPA {leader_val:+.3f}")
    return series


def main():
    semana_str = input("Semana (numero): ").strip()
    try:
        week = int(semana_str)
    except ValueError:
        raise SystemExit("Semana invalida.")

    print(f"Descargando pbp {SEASON}...")
    df = pd.read_csv(URL, low_memory=False, compression="infer")
    to_num(df, ["week", "epa", "sack"])

    d = df[df["week"] == week].copy()
    if d.empty:
        raise SystemExit(f"No hay jugadas para la semana {week}.")

    play_type   = pick_col(d, "play_type")
    posteam     = pick_col(d, "posteam")
    defteam     = pick_col(d, "defteam")
    passer      = pick_col(d, "passer", "passer_player_name")
    receiver    = pick_col(d, "receiver", "receiver_player_name")
    rusher      = pick_col(d, "rusher", "rusher_player_name")
    interception = pick_col(d, "interception_player_name")
    sack_p      = pick_col(d, "sack_player_name")
    kicker      = pick_col(d, "kicker_player_name", "kicker")
    punter      = pick_col(d, "punter_player_name", "punter")
    kr_ret      = pick_col(d, "kickoff_returner_player_name", "returner_player_name")
    pr_ret      = pick_col(d, "punt_returner_player_name",   "returner_player_name")

    of_credit  = calc_ataque(d, play_type, passer, receiver, rusher, posteam)
    def_credit = calc_defensa(d, interception, sack_p, defteam)
    st_credit  = calc_st(d, play_type, kr_ret, pr_ret, kicker, punter, posteam)

    print(f"\n========== LIDERES EPA — Semana {week} NFL {SEASON} ==========")
    of_series  = print_top("ATAQUE",             of_credit)
    def_series = print_top("DEFENSA",            def_credit)
    st_series  = print_top("EQUIPOS ESPECIALES", st_credit)

    show_top3 = input("\nMostrar top-3 por categoria? (s/n): ").strip().lower()
    if show_top3 == "s":
        for label, series in [("ATAQUE", of_series), ("DEFENSA", def_series), ("ST", st_series)]:
            if series is not None:
                print(f"\nTop-3 {label}:")
                print(series.head(3).apply(lambda v: f"{v:+.3f}").to_string())

if __name__ == "__main__":
    main()
