# MVPsSeason.py
# Líderes de EPA en TODA la temporada 2025:
# - Jugador de ATAQUE
# - Jugador de DEFENSA
# - ROOKIE de ATAQUE
# - ROOKIE de DEFENSA
# - Jugador de EQUIPOS ESPECIALES
#
# Fuentes online (nflverse):
#  - PBP 2025: play_by_play_2025.csv.gz
#  - Players: players.csv  (detección de rookies robusta + normalización de nombres)

import pandas as pd
import numpy as np
import re

URL_PBP = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.csv.gz"
URL_PLAYERS = "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv"

# ---------------- Helpers ----------------
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

def add_credit(store: dict, key, team, val):
    if key is None or (isinstance(key, float) and np.isnan(key)):
        return
    name = str(key)
    label = f"{name} ({team})" if team else name
    store[label] = store.get(label, 0.0) + float(val)

def split_credit(store: dict, keys, team, val):
    valid = [k for k in keys if k is not None and not (isinstance(k, float) and np.isnan(k))]
    if not valid:
        return
    share = float(val) / len(valid)
    for k in valid:
        add_credit(store, k, team, share)

def _norm_name(s: str) -> str:
    """Normaliza nombres para mejorar el emparejamiento por texto."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[.\,']", "", s)      # quita . , '
    s = re.sub(r"\s+", " ", s)        # colapsa espacios
    # elimina sufijos comunes
    for suf in [" jr", " sr", " ii", " iii", " iv", " v"]:
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
    return s

def build_rookie_sets(players_df: pd.DataFrame):
    """
    Devuelve (rookie_ids, rookie_names_norm) para 2025 de forma robusta.
    Estrategia:
      1) Usa 'first_season' / 'rookie_season' / 'rookie_year' / 'first_year' / 'debut_season' si existe.
      2) Si no, usa draft_year == 2025 si existe.
      3) Si tampoco, pero hay 'season' por jugador, toma primer año por jugador == 2025.
      4) Construye sets de IDs y de NOMBRES NORMALIZADOS.
    """
    df = players_df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    def getc(*opts):
        for o in opts:
            lo = o.lower()
            if lo in cols_lower:
                return cols_lower[lo]
        return None

    def has(c): 
        return c in df.columns

    # 1) columnas directas de rookie/first
    first_like = getc("first_season", "rookie_season", "rookie_year", "first_year", "debut_season")
    if first_like:
        df[first_like] = pd.to_numeric(df[first_like], errors="coerce")
        rook = df[df[first_like] == 2025].copy()
    else:
        # 2) draft_year
        draft_col = getc("draft_year")
        if draft_col:
            df[draft_col] = pd.to_numeric(df[draft_col], errors="coerce")
            rook = df[df[draft_col] == 2025].copy()
        else:
            # 3) derivar por 'season' si existe
            season_col = getc("season")
            id_col = getc("gsis_id", "nfl_id", "pfr_player_id", "pfr_id", "esb_id", "espn_id", "sportradar_id")
            name_col = getc("full_name", "display_name", "gsis_name", "player_name")
            if season_col:
                df[season_col] = pd.to_numeric(df[season_col], errors="coerce")
                if id_col:
                    firsty = df.groupby(id_col)[season_col].min().reset_index()
                    rook_ids_sel = set(firsty[firsty[season_col] == 2025][id_col].astype(str))
                    rook = df[df[id_col].astype(str).isin(rook_ids_sel)].copy()
                elif name_col:
                    firsty = df.groupby(name_col)[season_col].min().reset_index()
                    rook_names_sel = set(firsty[firsty[season_col] == 2025][name_col].astype(str))
                    rook = df[df[name_col].astype(str).isin(rook_names_sel)].copy()
                else:
                    rook = df.iloc[0:0].copy()
            else:
                rook = df.iloc[0:0].copy()

    # sets de IDs
    rookie_ids = set()
    for col in ["gsis_id", "nfl_id", "pfr_id", "pfr_player_id", "esb_id", "espn_id", "sportradar_id"]:
        if has(col):
            rookie_ids |= set(rook[col].dropna().astype(str).unique())

    # sets de nombres normalizados
    rookie_names_norm = set()
    for col in ["full_name", "display_name", "gsis_name", "player_name"]:
        if has(col):
            rookie_names_norm |= set(_norm_name(x) for x in rook[col].dropna().astype(str).unique())

    return rookie_ids, rookie_names_norm

def is_rookie(id_val, name_val, rookie_ids: set, rookie_names_norm: set):
    """True si el id está en rookies o el nombre normalizado coincide."""
    if id_val is not None and not (isinstance(id_val, float) and np.isnan(id_val)):
        if str(id_val) in rookie_ids:
            return True
    if name_val is not None and not (isinstance(name_val, float) and np.isnan(name_val)):
        if _norm_name(str(name_val)) in rookie_names_norm:
            return True
    return False

# ---------------- Main ----------------
def main():
    print("Descargando play-by-play 2025…")
    df = pd.read_csv(URL_PBP, low_memory=False, compression="infer")

    play_type = pick_col(df, "play_type")
    if play_type is None or "epa" not in df.columns:
        raise SystemExit("Faltan columnas esenciales (play_type o epa).")

    to_num(df, ["epa"])
    posteam = pick_col(df, "posteam")
    defteam = pick_col(df, "defteam")

    # IDs y nombres comunes (ataque)
    passer_id  = pick_col(df, "passer_player_id", "passer_id")
    passer_nm  = pick_col(df, "passer", "passer_player_name")

    receiver_id = pick_col(df, "receiver_player_id", "receiver_id")
    receiver_nm = pick_col(df, "receiver", "receiver_player_name")

    rusher_id  = pick_col(df, "rusher_player_id", "rusher_id")
    rusher_nm  = pick_col(df, "rusher", "rusher_player_name")

    # defensivos
    int_id = pick_col(df, "interception_player_id")
    int_nm = pick_col(df, "interception_player_name", "interception_player")

    sack_id = pick_col(df, "sack_player_id")
    sack_nm = pick_col(df, "sack_player_name")

    tfl1_id = pick_col(df, "tackle_for_loss_1_player_id")
    tfl1_nm = pick_col(df, "tackle_for_loss_1_player_name", "tfl_player_name")
    tfl2_id = pick_col(df, "tackle_for_loss_2_player_id")
    tfl2_nm = pick_col(df, "tackle_for_loss_2_player_name")

    ff1_id = pick_col(df, "forced_fumble_player_1_player_id")
    ff1_nm = pick_col(df, "forced_fumble_player_1_player_name")
    ff2_id = pick_col(df, "forced_fumble_player_2_player_id")
    ff2_nm = pick_col(df, "forced_fumble_player_2_player_name")

    pd1_id = pick_col(df, "pass_defensed_1_player_id")
    pd1_nm = pick_col(df, "pass_defensed_1_player_name", "pass_defense_1_player_name")
    pd2_id = pick_col(df, "pass_defensed_2_player_id")
    pd2_nm = pick_col(df, "pass_defensed_2_player_name", "pass_defense_2_player_name")

    # equipos especiales
    kicker_nm = pick_col(df, "kicker_player_name", "kicker")
    punter_nm = pick_col(df, "punter_player_name", "punter")
    kr_nm     = pick_col(df, "kickoff_returner_player_name", "returner_player_name", "returner")
    pr_nm     = pick_col(df, "punt_returner_player_name", "returner_player_name", "returner")

    # ---------------- Rookies ----------------
    print("Descargando players.csv para detectar rookies…")
    players = pd.read_csv(URL_PLAYERS, low_memory=False)
    rookie_ids, rookie_names_norm = build_rookie_sets(players)

    # ---------------- ACUMULADORES ----------------
    offense = {}           # nombre (con equipo) -> EPA
    defense = {}
    st = {}
    offense_rookie = {}
    defense_rookie = {}

    # ---------------- ATAQUE ----------------
    data_off = df[df["epa"].notna()].copy()

    # PASE: 50% pasador + 50% receptor (si receptor falta, 100% al pasador)
    passes = data_off[data_off[play_type] == "pass"].copy()
    if not passes.empty:
        for _, r in passes.iterrows():
            epa = float(r["epa"])
            team = r.get(posteam, None)

            qb_id = r.get(passer_id) if passer_id else None
            qb_nm_val = r.get(passer_nm) if passer_nm else None
            rc_id = r.get(receiver_id) if receiver_id else None
            rc_nm_val = r.get(receiver_nm) if receiver_nm else None

            if qb_nm_val is not None:
                add_credit(offense, qb_nm_val, team, epa * 0.5)
                if is_rookie(qb_id, qb_nm_val, rookie_ids, rookie_names_norm):
                    add_credit(offense_rookie, qb_nm_val, team, epa * 0.5)
            if rc_nm_val is not None:
                add_credit(offense, rc_nm_val, team, epa * 0.5)
                if is_rookie(rc_id, rc_nm_val, rookie_ids, rookie_names_norm):
                    add_credit(offense_rookie, rc_nm_val, team, epa * 0.5)
            else:
                if qb_nm_val is not None:
                    add_credit(offense, qb_nm_val, team, epa * 0.5)
                    if is_rookie(qb_id, qb_nm_val, rookie_ids, rookie_names_norm):
                        add_credit(offense_rookie, qb_nm_val, team, epa * 0.5)

    # CARRERA: 100% al portador (incluye QB runs)
    runs = data_off[data_off[play_type] == "run"].copy()
    if not runs.empty:
        for _, r in runs.iterrows():
            epa = float(r["epa"])
            team = r.get(posteam, None)
            rus_id_val = r.get(rusher_id) if rusher_id else None
            rus_nm_val = r.get(rusher_nm) if rusher_nm else None
            if rus_nm_val is not None:
                add_credit(offense, rus_nm_val, team, epa)
                if is_rookie(rus_id_val, rus_nm_val, rookie_ids, rookie_names_norm):
                    add_credit(offense_rookie, rus_nm_val, team, epa)

    # ---------------- DEFENSA ----------------
    data_def = df[df["epa"].notna()].copy()

    # Intercepciones
    if int_nm or int_id:
        mask = pd.Series(False, index=data_def.index)
        if int_nm: mask |= data_def[int_nm].notna()
        if int_id: mask |= data_def[int_id].notna()
        rows = data_def[mask]
        for _, r in rows.iterrows():
            team = r.get(defteam, None)
            defenders = []
            if int_nm and pd.notna(r.get(int_nm)): defenders.append(r.get(int_nm))
            if defenders:
                split_credit(defense, defenders, team, -r["epa"])
                for dnm in defenders:
                    if is_rookie(None, dnm, rookie_ids, rookie_names_norm):
                        add_credit(defense_rookie, dnm, team, -r["epa"])

    # Sacks
    if "sack" in df.columns and (sack_nm or sack_id):
        rows = data_def[(data_def["sack"] == 1)]
        rows = rows[(sack_nm and rows.get(sack_nm, pd.Series(index=rows.index)).notna()) |
                    (sack_id and rows.get(sack_id, pd.Series(index=rows.index)).notna())]
        for _, r in rows.iterrows():
            team = r.get(defteam, None)
            defenders = []
            if sack_nm and pd.notna(r.get(sack_nm)): defenders.append(r.get(sack_nm))
            if defenders:
                split_credit(defense, defenders, team, -r["epa"])
                for dnm in defenders:
                    if is_rookie(None, dnm, rookie_ids, rookie_names_norm):
                        add_credit(defense_rookie, dnm, team, -r["epa"])

    # TFL (uno o dos)
    if (tfl1_nm or tfl1_id) or (tfl2_nm or tfl2_id):
        mask = pd.Series(False, index=data_def.index)
        if tfl1_nm: mask |= data_def[tfl1_nm].notna()
        if tfl2_nm: mask |= data_def[tfl2_nm].notna()
        rows = data_def[mask]
        for _, r in rows.iterrows():
            team = r.get(defteam, None)
            defenders = []
            if tfl1_nm and pd.notna(r.get(tfl1_nm)): defenders.append(r.get(tfl1_nm))
            if tfl2_nm and pd.notna(r.get(tfl2_nm)): defenders.append(r.get(tfl2_nm))
            if defenders:
                split_credit(defense, defenders, team, -r["epa"])
                for dnm in defenders:
                    if is_rookie(None, dnm, rookie_ids, rookie_names_norm):
                        add_credit(defense_rookie, dnm, team, -r["epa"])

    # Fumbles forzados
    if (ff1_nm or ff1_id) or (ff2_nm or ff2_id):
        mask = pd.Series(False, index=data_def.index)
        if ff1_nm: mask |= data_def[ff1_nm].notna()
        if ff2_nm: mask |= data_def[ff2_nm].notna()
        rows = data_def[mask]
        for _, r in rows.iterrows():
            team = r.get(defteam, None)
            defenders = []
            if ff1_nm and pd.notna(r.get(ff1_nm)): defenders.append(r.get(ff1_nm))
            if ff2_nm and pd.notna(r.get(ff2_nm)): defenders.append(r.get(ff2_nm))
            if defenders:
                split_credit(defense, defenders, team, -r["epa"])
                for dnm in defenders:
                    if is_rookie(None, dnm, rookie_ids, rookie_names_norm):
                        add_credit(defense_rookie, dnm, team, -r["epa"])

    # Passes defendidos
    if (pd1_nm or pd1_id) or (pd2_nm or pd2_id):
        mask = pd.Series(False, index=data_def.index)
        if pd1_nm: mask |= data_def[pd1_nm].notna()
        if pd2_nm: mask |= data_def[pd2_nm].notna()
        rows = data_def[mask]
        for _, r in rows.iterrows():
            team = r.get(defteam, None)
            defenders = []
            if pd1_nm and pd.notna(r.get(pd1_nm)): defenders.append(r.get(pd1_nm))
            if pd2_nm and pd.notna(r.get(pd2_nm)): defenders.append(r.get(pd2_nm))
            if defenders:
                split_credit(defense, defenders, team, -r["epa"])
                for dnm in defenders:
                    if is_rookie(None, dnm, rookie_ids, rookie_names_norm):
                        add_credit(defense_rookie, dnm, team, -r["epa"])

    # ---------------- EQUIPOS ESPECIALES ----------------
    data_st = df[df["epa"].notna()].copy()

    # Returns
    if kr_nm:
        rows = data_st[(data_st[play_type] == "kickoff_return") & data_st[kr_nm].notna()]
        for _, r in rows.iterrows():
            add_credit(st, r[kr_nm], r.get(posteam), r["epa"])
    if pr_nm:
        rows = data_st[(data_st[play_type] == "punt_return") & data_st[pr_nm].notna()]
        for _, r in rows.iterrows():
            add_credit(st, r[pr_nm], r.get(posteam), r["epa"])

    # FG / XP → kicker
    if kicker_nm:
        rows = data_st[(data_st[play_type] == "field_goal") & data_st[kicker_nm].notna()]
        for _, r in rows.iterrows():
            add_credit(st, r[kicker_nm], r.get(posteam), r["epa"])
        rows = data_st[(data_st[play_type] == "extra_point") & data_st[kicker_nm].notna()]
        for _, r in rows.iterrows():
            add_credit(st, r[kicker_nm], r.get(posteam), r["epa"])

    # Punts → punter
    if punter_nm:
        rows = data_st[(data_st[play_type] == "punt") & data_st[punter_nm].notna()]
        for _, r in rows.iterrows():
            add_credit(st, r[punter_nm], r.get(posteam), r["epa"])

    # ---------------- RESULTADOS ----------------
    def top1(dct):
        if not dct:
            return None, None
        s = pd.Series(dct).sort_values(ascending=False)
        return s.index[0], s.iloc[0]

    of_name, of_val       = top1(offense)
    def_name, def_val     = top1(defense)
    of_r_name, of_r_val   = top1(offense_rookie)
    def_r_name, def_r_val = top1(defense_rookie)
    st_name, st_val       = top1(st)

    print("\n=========== LÍDERES EPA — TEMPORADA 2025 ===========")
    if of_name:
        print(f"\nATAQUE → {of_name}:  EPA total {of_val:+.3f}")
    else:
        print("\nATAQUE → sin datos")

    if def_name:
        print(f"DEFENSA → {def_name}:  EPA 'defensivo' {def_val:+.3f}")
    else:
        print("DEFENSA → sin datos")

    if of_r_name:
        print(f"ROOKIE ATAQUE → {of_r_name}:  EPA total {of_r_val:+.3f}")
    else:
        print("ROOKIE ATAQUE → sin datos")

    if def_r_name:
        print(f"ROOKIE DEFENSA → {def_r_name}:  EPA 'defensivo' {def_r_val:+.3f}")
    else:
        print("ROOKIE DEFENSA → sin datos")

    if st_name:
        print(f"EQUIPOS ESPECIALES → {st_name}:  EPA ST {st_val:+.3f}")
    else:
        print("EQUIPOS ESPECIALES → sin datos")

    # Top3 opcional por categoría
    show_top3 = input("\n¿Mostrar top-3 por categoría? (s/n): ").strip().lower()
    if show_top3 == "s":
        def show_top(dct, title):
            if dct:
                print(f"\nTop-3 {title}:")
                s = pd.Series(dct).sort_values(ascending=False).head(3)
                print(s.apply(lambda v: f"{v:+.3f}").to_string())
        show_top(offense, "ATAQUE")
        show_top(defense, "DEFENSA")
        show_top(offense_rookie, "ROOKIE ATAQUE")
        show_top(defense_rookie, "ROOKIE DEFENSA")
        show_top(st, "EQUIPOS ESPECIALES")

if __name__ == "__main__":
    main()
