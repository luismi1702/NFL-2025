# MVPsSeason.py
# Líderes de EPA en TODA la temporada:
# - Jugador de ATAQUE
# - Jugador de DEFENSA
# - ROOKIE de ATAQUE
# - ROOKIE de DEFENSA
# - Jugador de EQUIPOS ESPECIALES
#
# Fuentes online (nflverse):
#  - PBP: play_by_play_{SEASON}.csv.gz
#  - Players: players.csv  (detección de rookies robusta + normalización de nombres)

import pandas as pd
import numpy as np
import re

# === Config ===
SEASON = 2025
URL_PBP     = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"
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

def make_key(name_col, team_col):
    """Combina nombre y equipo en 'Nombre (TEAM)'."""
    return name_col.str.strip() + " (" + team_col.fillna("?") + ")"

def _norm_name(s: str) -> str:
    """Normaliza nombres para mejorar el emparejamiento por texto."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[.\,']", "", s)
    s = re.sub(r"\s+", " ", s)
    for suf in [" jr", " sr", " ii", " iii", " iv", " v"]:
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
    return s

def build_rookie_sets(players_df: pd.DataFrame):
    """
    Devuelve (rookie_ids, rookie_names_norm) para SEASON de forma robusta.
    Estrategia en cascada:
      1) first_season / rookie_season / rookie_year / first_year / debut_season
      2) draft_year == SEASON
      3) primer año por jugador == SEASON (via 'season' column)
    """
    df = players_df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    def getc(*opts):
        for o in opts:
            if o.lower() in cols_lower:
                return cols_lower[o.lower()]
        return None

    first_like = getc("first_season", "rookie_season", "rookie_year", "first_year", "debut_season")
    if first_like:
        df[first_like] = pd.to_numeric(df[first_like], errors="coerce")
        rook = df[df[first_like] == SEASON].copy()
    else:
        draft_col = getc("draft_year")
        if draft_col:
            df[draft_col] = pd.to_numeric(df[draft_col], errors="coerce")
            rook = df[df[draft_col] == SEASON].copy()
        else:
            season_col = getc("season")
            id_col = getc("gsis_id", "nfl_id", "pfr_player_id", "pfr_id", "esb_id", "espn_id", "sportradar_id")
            name_col = getc("full_name", "display_name", "gsis_name", "player_name")
            if season_col:
                df[season_col] = pd.to_numeric(df[season_col], errors="coerce")
                if id_col:
                    firsty = df.groupby(id_col)[season_col].min().reset_index()
                    rook_ids_sel = set(firsty[firsty[season_col] == SEASON][id_col].astype(str))
                    rook = df[df[id_col].astype(str).isin(rook_ids_sel)].copy()
                elif name_col:
                    firsty = df.groupby(name_col)[season_col].min().reset_index()
                    rook_names_sel = set(firsty[firsty[season_col] == SEASON][name_col].astype(str))
                    rook = df[df[name_col].astype(str).isin(rook_names_sel)].copy()
                else:
                    rook = df.iloc[0:0].copy()
            else:
                rook = df.iloc[0:0].copy()

    rookie_ids = set()
    for col in ["gsis_id", "nfl_id", "pfr_id", "pfr_player_id", "esb_id", "espn_id", "sportradar_id"]:
        if col in df.columns:
            rookie_ids |= set(rook[col].dropna().astype(str).unique())

    rookie_names_norm = set()
    for col in ["full_name", "display_name", "gsis_name", "player_name"]:
        if col in df.columns:
            rookie_names_norm |= set(_norm_name(x) for x in rook[col].dropna().astype(str).unique())

    return rookie_ids, rookie_names_norm

def is_rookie_name(name_val, rookie_names_norm: set) -> bool:
    if not isinstance(name_val, str) or not name_val.strip():
        return False
    return _norm_name(name_val) in rookie_names_norm


# ---------------- Ataque (vectorizado) ----------------
def calc_ataque(d, play_type, passer, receiver, rusher, posteam, rookie_names_norm):
    credits = {}
    rook_credits = {}

    # -- Pases: QB 50% base siempre + QB 50% extra si no hay receptor --
    if passer and play_type:
        passes = d[(d[play_type] == "pass") & d["epa"].notna() & d[passer].notna()].copy()
        if not passes.empty:
            passes["_key_qb"] = make_key(passes[passer], passes[posteam] if posteam else pd.Series("", index=passes.index))

            qb_base = passes.groupby("_key_qb")["epa"].sum() * 0.5
            for k, v in qb_base.items():
                credits[k] = credits.get(k, 0.0) + v

            no_rec = passes[passes[receiver].isna()] if receiver else passes
            qb_extra = no_rec.groupby("_key_qb")["epa"].sum() * 0.5
            for k, v in qb_extra.items():
                credits[k] = credits.get(k, 0.0) + v

            # rookies QB
            passes["_is_rook"] = passes[passer].apply(lambda n: is_rookie_name(n, rookie_names_norm))
            rook_passes = passes[passes["_is_rook"]]
            if not rook_passes.empty:
                rk_base = rook_passes.groupby("_key_qb")["epa"].sum() * 0.5
                for k, v in rk_base.items():
                    rook_credits[k] = rook_credits.get(k, 0.0) + v
                no_rec_rk = rook_passes[rook_passes[receiver].isna()] if receiver else rook_passes
                rk_extra = no_rec_rk.groupby("_key_qb")["epa"].sum() * 0.5
                for k, v in rk_extra.items():
                    rook_credits[k] = rook_credits.get(k, 0.0) + v

            # Receptor 50%
            if receiver:
                rec_plays = passes[passes[receiver].notna()].copy()
                if not rec_plays.empty:
                    rec_plays["_key_rec"] = make_key(rec_plays[receiver], rec_plays[posteam] if posteam else pd.Series("", index=rec_plays.index))
                    rec_total = rec_plays.groupby("_key_rec")["epa"].sum() * 0.5
                    for k, v in rec_total.items():
                        credits[k] = credits.get(k, 0.0) + v

                    rec_plays["_is_rook"] = rec_plays[receiver].apply(lambda n: is_rookie_name(n, rookie_names_norm))
                    rook_rec = rec_plays[rec_plays["_is_rook"]]
                    if not rook_rec.empty:
                        rk_rec = rook_rec.groupby("_key_rec")["epa"].sum() * 0.5
                        for k, v in rk_rec.items():
                            rook_credits[k] = rook_credits.get(k, 0.0) + v

    # -- Carreras: 100% al portador --
    if rusher and play_type:
        runs = d[(d[play_type] == "run") & d["epa"].notna() & d[rusher].notna()].copy()
        if not runs.empty:
            runs["_key"] = make_key(runs[rusher], runs[posteam] if posteam else pd.Series("", index=runs.index))
            run_total = runs.groupby("_key")["epa"].sum()
            for k, v in run_total.items():
                credits[k] = credits.get(k, 0.0) + v

            runs["_is_rook"] = runs[rusher].apply(lambda n: is_rookie_name(n, rookie_names_norm))
            rook_runs = runs[runs["_is_rook"]]
            if not rook_runs.empty:
                rk_run = rook_runs.groupby("_key")["epa"].sum()
                for k, v in rk_run.items():
                    rook_credits[k] = rook_credits.get(k, 0.0) + v

    return credits, rook_credits


# ---------------- Defensa (vectorizado) ----------------
def _add_def_credits(sub, name_col1, name_col2, team_col, credits, rook_credits, rookie_names_norm, multiplier=-1.0):
    """Para jugadas defensivas con hasta 2 jugadores, acumula crédito dividido."""
    if sub.empty:
        return

    # jugadas con solo nm1
    if name_col1 and name_col2:
        only1 = sub[sub[name_col1].notna() & sub[name_col2].isna()].copy()
        only2 = sub[sub[name_col2].notna() & sub[name_col1].isna()].copy()
        both  = sub[sub[name_col1].notna() & sub[name_col2].notna()].copy()
    elif name_col1:
        only1 = sub[sub[name_col1].notna()].copy()
        only2 = pd.DataFrame()
        both  = pd.DataFrame()
    else:
        return

    def _accum(rows, col, factor):
        if rows.empty or not col:
            return
        rows = rows.copy()
        rows["_key"] = make_key(rows[col], rows[team_col] if team_col else pd.Series("?", index=rows.index))
        totals = rows.groupby("_key")["epa"].sum() * multiplier * factor
        for k, v in totals.items():
            credits[k] = credits.get(k, 0.0) + v
        # rookies
        rows["_is_rook"] = rows[col].apply(lambda n: is_rookie_name(n, rookie_names_norm))
        rook_rows = rows[rows["_is_rook"]]
        if not rook_rows.empty:
            rk = rook_rows.groupby("_key")["epa"].sum() * multiplier * factor
            for k, v in rk.items():
                rook_credits[k] = rook_credits.get(k, 0.0) + v

    _accum(only1, name_col1, 1.0)
    _accum(only2, name_col2, 1.0)
    _accum(both,  name_col1, 0.5)
    _accum(both,  name_col2, 0.5)


def calc_defensa(d, int_nm, sack_nm, tfl1_nm, tfl2_nm, ff1_nm, ff2_nm, pd1_nm, pd2_nm, defteam, rookie_names_norm):
    credits = {}
    rook_credits = {}

    base = d[d["epa"].notna()]

    # Intercepciones
    if int_nm:
        sub = base[base[int_nm].notna()]
        _add_def_credits(sub, int_nm, None, defteam, credits, rook_credits, rookie_names_norm)

    # Sacks
    if sack_nm and "sack" in d.columns:
        sub = base[(base["sack"] == 1) & base[sack_nm].notna()]
        _add_def_credits(sub, sack_nm, None, defteam, credits, rook_credits, rookie_names_norm)

    # TFL
    if tfl1_nm or tfl2_nm:
        mask = pd.Series(False, index=base.index)
        if tfl1_nm: mask |= base[tfl1_nm].notna()
        if tfl2_nm: mask |= base[tfl2_nm].notna()
        _add_def_credits(base[mask], tfl1_nm, tfl2_nm, defteam, credits, rook_credits, rookie_names_norm)

    # Fumbles forzados
    if ff1_nm or ff2_nm:
        mask = pd.Series(False, index=base.index)
        if ff1_nm: mask |= base[ff1_nm].notna()
        if ff2_nm: mask |= base[ff2_nm].notna()
        _add_def_credits(base[mask], ff1_nm, ff2_nm, defteam, credits, rook_credits, rookie_names_norm)

    # Passes defendidos
    if pd1_nm or pd2_nm:
        mask = pd.Series(False, index=base.index)
        if pd1_nm: mask |= base[pd1_nm].notna()
        if pd2_nm: mask |= base[pd2_nm].notna()
        _add_def_credits(base[mask], pd1_nm, pd2_nm, defteam, credits, rook_credits, rookie_names_norm)

    return credits, rook_credits


# ---------------- Equipos Especiales (vectorizado) ----------------
def calc_st(d, play_type, kr_nm, pr_nm, kicker_nm, punter_nm, posteam):
    credits = {}

    def _add(plays, name_col, team_col):
        if not name_col or plays.empty:
            return
        sub = plays[plays[name_col].notna()].copy()
        if sub.empty:
            return
        sub["_key"] = make_key(sub[name_col], sub[team_col] if team_col else pd.Series("?", index=sub.index))
        totals = sub.groupby("_key")["epa"].sum()
        for k, v in totals.items():
            credits[k] = credits.get(k, 0.0) + v

    if play_type:
        base = d[d["epa"].notna()]
        _add(base[base[play_type] == "kickoff_return"],  kr_nm,     posteam)
        _add(base[base[play_type] == "punt_return"],     pr_nm,     posteam)
        _add(base[base[play_type] == "field_goal"],      kicker_nm, posteam)
        _add(base[base[play_type] == "extra_point"],     kicker_nm, posteam)
        _add(base[base[play_type] == "punt"],            punter_nm, posteam)

    return credits


# ---------------- Main ----------------
def main():
    print(f"Descargando play-by-play {SEASON}...")
    df = pd.read_csv(URL_PBP, low_memory=False, compression="infer")

    play_type = pick_col(df, "play_type")
    if play_type is None or "epa" not in df.columns:
        raise SystemExit("Faltan columnas esenciales (play_type o epa).")

    to_num(df, ["epa", "sack"])
    posteam = pick_col(df, "posteam")
    defteam = pick_col(df, "defteam")

    passer_nm   = pick_col(df, "passer", "passer_player_name")
    receiver_nm = pick_col(df, "receiver", "receiver_player_name")
    rusher_nm   = pick_col(df, "rusher", "rusher_player_name")

    int_nm  = pick_col(df, "interception_player_name", "interception_player")
    sack_nm = pick_col(df, "sack_player_name")
    tfl1_nm = pick_col(df, "tackle_for_loss_1_player_name", "tfl_player_name")
    tfl2_nm = pick_col(df, "tackle_for_loss_2_player_name")
    ff1_nm  = pick_col(df, "forced_fumble_player_1_player_name")
    ff2_nm  = pick_col(df, "forced_fumble_player_2_player_name")
    pd1_nm  = pick_col(df, "pass_defensed_1_player_name", "pass_defense_1_player_name")
    pd2_nm  = pick_col(df, "pass_defensed_2_player_name", "pass_defense_2_player_name")

    kicker_nm = pick_col(df, "kicker_player_name", "kicker")
    punter_nm = pick_col(df, "punter_player_name", "punter")
    kr_nm     = pick_col(df, "kickoff_returner_player_name", "returner_player_name")
    pr_nm     = pick_col(df, "punt_returner_player_name",   "returner_player_name")

    # Rookies
    print("Descargando players.csv para detectar rookies...")
    players = pd.read_csv(URL_PLAYERS, low_memory=False)
    _, rookie_names_norm = build_rookie_sets(players)
    print(f"Rookies detectados: {len(rookie_names_norm)}")

    # Calcular
    of_credit, of_rook   = calc_ataque(df, play_type, passer_nm, receiver_nm, rusher_nm, posteam, rookie_names_norm)
    def_credit, def_rook = calc_defensa(df, int_nm, sack_nm, tfl1_nm, tfl2_nm, ff1_nm, ff2_nm, pd1_nm, pd2_nm, defteam, rookie_names_norm)
    st_credit            = calc_st(df, play_type, kr_nm, pr_nm, kicker_nm, punter_nm, posteam)

    def top1(dct):
        if not dct:
            return None, None
        s = pd.Series(dct).sort_values(ascending=False)
        return s.index[0], s.iloc[0]

    of_name,   of_val   = top1(of_credit)
    def_name,  def_val  = top1(def_credit)
    of_r_name, of_r_val = top1(of_rook)
    dr_name,   dr_val   = top1(def_rook)
    st_name,   st_val   = top1(st_credit)

    print(f"\n========== LIDERES EPA — TEMPORADA {SEASON} ==========")
    print(f"  ATAQUE             : {of_name}  EPA {of_val:+.3f}"   if of_name   else "  ATAQUE             : sin datos")
    print(f"  DEFENSA            : {def_name}  EPA {def_val:+.3f}" if def_name  else "  DEFENSA            : sin datos")
    print(f"  ROOKIE ATAQUE      : {of_r_name}  EPA {of_r_val:+.3f}" if of_r_name else "  ROOKIE ATAQUE      : sin datos")
    print(f"  ROOKIE DEFENSA     : {dr_name}  EPA {dr_val:+.3f}"   if dr_name   else "  ROOKIE DEFENSA     : sin datos")
    print(f"  EQUIPOS ESPECIALES : {st_name}  EPA {st_val:+.3f}"   if st_name   else "  EQUIPOS ESPECIALES : sin datos")

    show_top3 = input("\nMostrar top-3 por categoria? (s/n): ").strip().lower()
    if show_top3 == "s":
        def show_top(dct, title):
            if dct:
                print(f"\nTop-3 {title}:")
                s = pd.Series(dct).sort_values(ascending=False).head(3)
                print(s.apply(lambda v: f"{v:+.3f}").to_string())
        show_top(of_credit,  "ATAQUE")
        show_top(def_credit, "DEFENSA")
        show_top(of_rook,    "ROOKIE ATAQUE")
        show_top(def_rook,   "ROOKIE DEFENSA")
        show_top(st_credit,  "EQUIPOS ESPECIALES")

if __name__ == "__main__":
    main()
