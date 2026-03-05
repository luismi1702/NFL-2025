# leaders_semana_of_def_st.py
# Líder semanal de EPA (ataque, defensa, equipos especiales)
# Añadido: muestra el EQUIPO de cada jugador (usando posteam o defteam).

import pandas as pd
import numpy as np

URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.csv.gz"

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

def add_credit(agg_dict, name, team, value):
    if not name or pd.isna(name):
        return
    key = f"{name} ({team})" if team else str(name)
    agg_dict[key] = agg_dict.get(key, 0.0) + float(value)

def split_credit(agg_dict, names, team, value):
    valid = [n for n in names if n and not pd.isna(n)]
    if not valid:
        return
    share = float(value) / len(valid)
    for n in valid:
        add_credit(agg_dict, n, team, share)

def main():
    semana_str = input("Semana (número): ").strip()
    try:
        week = int(semana_str)
    except:
        raise SystemExit("Semana inválida.")

    print("Descargando pbp 2025…")
    df = pd.read_csv(URL, low_memory=False, compression="infer")

    if "week" not in df.columns:
        raise SystemExit("El dataset no tiene columna 'week'.")

    to_num(df, ["week", "epa"])
    d = df[df["week"] == week].copy()
    if d.empty:
        raise SystemExit(f"No hay jugadas para la semana {week}.")

    play_type = pick_col(d, "play_type")
    posteam = pick_col(d, "posteam")
    defteam = pick_col(d, "defteam")

    passer = pick_col(d, "passer", "passer_player_name")
    receiver = pick_col(d, "receiver", "receiver_player_name")
    rusher = pick_col(d, "rusher", "rusher_player_name")

    interception = pick_col(d, "interception_player_name")
    sack_p = pick_col(d, "sack_player_name")
    kicker = pick_col(d, "kicker_player_name", "kicker")
    punter = pick_col(d, "punter_player_name", "punter")
    kr_ret = pick_col(d, "kickoff_returner_player_name", "returner_player_name", "returner")
    pr_ret = pick_col(d, "punt_returner_player_name", "returner_player_name", "returner")

    # =================== ATAQUE ===================
    of_credit = {}

    # PASES (50% QB + 50% receptor)
    if play_type and passer and "epa" in d.columns:
        passes = d[(d[play_type] == "pass") & d["epa"].notna()].copy()
        if not passes.empty:
            for _, r in passes.iterrows():
                epa = r["epa"]
                team = r.get(posteam, None)
                qb = r.get(passer)
                rec = r.get(receiver)
                if pd.notna(qb):
                    add_credit(of_credit, qb, team, epa * 0.5)
                if pd.notna(rec):
                    add_credit(of_credit, rec, team, epa * 0.5)
                elif pd.notna(qb):
                    add_credit(of_credit, qb, team, epa * 0.5)

    # CARRERAS (100% al portador)
    if play_type and rusher and "epa" in d.columns:
        runs = d[(d[play_type] == "run") & d["epa"].notna()]
        for _, r in runs.iterrows():
            epa = r["epa"]
            team = r.get(posteam, None)
            rus = r.get(rusher)
            if pd.notna(rus):
                add_credit(of_credit, rus, team, epa)

    of_leader_name, of_leader_val = (None, None)
    if of_credit:
        of_series = pd.Series(of_credit).sort_values(ascending=False)
        of_leader_name = of_series.index[0]
        of_leader_val = of_series.iloc[0]

    # =================== DEFENSA ===================
    def_credit = {}

    # Intercepciones
    if interception:
        ints = d[(d[interception].notna()) & d["epa"].notna()]
        for _, r in ints.iterrows():
            split_credit(def_credit, [r[interception]], r.get(defteam), -r["epa"])

    # Sacks
    if sack_p and "sack" in d.columns:
        sacks = d[(d["sack"] == 1) & d["epa"].notna() & d[sack_p].notna()]
        for _, r in sacks.iterrows():
            split_credit(def_credit, [r[sack_p]], r.get(defteam), -r["epa"])

    def_leader_name, def_leader_val = (None, None)
    if def_credit:
        def_series = pd.Series(def_credit).sort_values(ascending=False)
        def_leader_name = def_series.index[0]
        def_leader_val = def_series.iloc[0]

    # =================== EQUIPOS ESPECIALES ===================
    st_credit = {}

    if play_type and kr_ret:
        kret = d[(d[play_type] == "kickoff_return") & d["epa"].notna() & d[kr_ret].notna()]
        for _, r in kret.iterrows():
            add_credit(st_credit, r[kr_ret], r.get(posteam), r["epa"])

    if play_type and pr_ret:
        pret = d[(d[play_type] == "punt_return") & d["epa"].notna() & d[pr_ret].notna()]
        for _, r in pret.iterrows():
            add_credit(st_credit, r[pr_ret], r.get(posteam), r["epa"])

    if play_type and kicker:
        fgs = d[(d[play_type] == "field_goal") & d["epa"].notna() & d[kicker].notna()]
        for _, r in fgs.iterrows():
            add_credit(st_credit, r[kicker], r.get(posteam), r["epa"])
        xps = d[(d[play_type] == "extra_point") & d["epa"].notna() & d[kicker].notna()]
        for _, r in xps.iterrows():
            add_credit(st_credit, r[kicker], r.get(posteam), r["epa"])

    if play_type and punter:
        punts = d[(d[play_type] == "punt") & d["epa"].notna() & d[punter].notna()]
        for _, r in punts.iterrows():
            add_credit(st_credit, r[punter], r.get(posteam), r["epa"])

    st_leader_name, st_leader_val = (None, None)
    if st_credit:
        st_series = pd.Series(st_credit).sort_values(ascending=False)
        st_leader_name = st_series.index[0]
        st_leader_val = st_series.iloc[0]

    # =================== RESULTADOS ===================
    print(f"\n=========== LÍDERES EPA — Semana {week} ===========")

    if of_leader_name:
        print(f"\nATAQUE → {of_leader_name}:  EPA total {of_leader_val:+.3f}")
    else:
        print("\nATAQUE → sin datos")

    if def_leader_name:
        print(f"DEFENSA → {def_leader_name}:  EPA 'defensivo' {def_leader_val:+.3f}")
    else:
        print("DEFENSA → sin datos")

    if st_leader_name:
        print(f"EQUIPOS ESPECIALES → {st_leader_name}:  EPA ST {st_leader_val:+.3f}")
    else:
        print("EQUIPOS ESPECIALES → sin datos")

    # top-3 opcional
    show_top3 = input("\n¿Mostrar top-3 por categoría? (s/n): ").strip().lower()
    if show_top3 == "s":
        if of_credit:
            print("\nTop-3 ATAQUE:")
            print(pd.Series(of_credit).sort_values(ascending=False).head(3).apply(lambda v: f"{v:+.3f}").to_string())
        if def_credit:
            print("\nTop-3 DEFENSA:")
            print(pd.Series(def_credit).sort_values(ascending=False).head(3).apply(lambda v: f"{v:+.3f}").to_string())
        if st_credit:
            print("\nTop-3 ST:")
            print(pd.Series(st_credit).sort_values(ascending=False).head(3).apply(lambda v: f"{v:+.3f}").to_string())

if __name__ == "__main__":
    main()
