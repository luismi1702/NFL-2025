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
from pbp_loader import cargar_pbp

# === Config ===
SEASON = None   # None = auto-detectar última temporada
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
    Devuelve (rookie_ids, rookie_names_norm) para SEASON.
    Estrategia en cascada:
      1) rookie_year / entry_year / first_season / rookie_season / first_year / debut_season == SEASON
      2) years_exp == 0  (rookies en activo esta temporada)
      3) draft_year == SEASON
    Los nombres normalizados incluyen tanto nombres completos como el formato
    abreviado del PBP (F.Apellido) generado desde first_name + last_name.
    """
    df = players_df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    def getc(*opts):
        for o in opts:
            if o.lower() in cols_lower:
                return cols_lower[o.lower()]
        return None

    # Estrategia 1: columna de año de rookie/entrada
    first_like = getc("rookie_year", "entry_year", "first_season",
                      "rookie_season", "first_year", "debut_season")
    if first_like:
        df[first_like] = pd.to_numeric(df[first_like], errors="coerce")
        rook = df[df[first_like] == SEASON].copy()
    else:
        # Estrategia 2: years_exp == 0
        exp_col = getc("years_exp")
        if exp_col:
            df[exp_col] = pd.to_numeric(df[exp_col], errors="coerce")
            rook = df[df[exp_col] == 0].copy()
        else:
            # Estrategia 3: draft_year
            draft_col = getc("draft_year")
            if draft_col:
                df[draft_col] = pd.to_numeric(df[draft_col], errors="coerce")
                rook = df[df[draft_col] == SEASON].copy()
            else:
                rook = df.iloc[0:0].copy()

    rookie_ids = set()
    for col in ["gsis_id", "nfl_id", "pfr_id", "pfr_player_id", "esb_id", "espn_id", "sportradar_id"]:
        if col in df.columns:
            rookie_ids |= set(rook[col].dropna().astype(str).unique())

    rookie_names_norm = set()
    # Nombres completos/display
    for col in ["full_name", "display_name", "gsis_name", "player_name", "football_name"]:
        if col in rook.columns:
            rookie_names_norm |= set(_norm_name(x) for x in rook[col].dropna().astype(str).unique())
    # Nombres abreviados estilo PBP: "F.Apellido"
    fn_col = getc("first_name")
    ln_col = getc("last_name")
    if fn_col and ln_col and fn_col in rook.columns and ln_col in rook.columns:
        abbrevs = (rook[fn_col].str.strip().str[:1] + "." +
                   rook[ln_col].str.strip()).dropna()
        rookie_names_norm |= set(_norm_name(x) for x in abbrevs.unique())

    return rookie_ids, rookie_names_norm

def is_rookie_name(name_val, rookie_names_norm: set) -> bool:
    if not isinstance(name_val, str) or not name_val.strip():
        return False
    return _norm_name(name_val) in rookie_names_norm


def build_pbp_rookie_names(pbp_df: pd.DataFrame, rookie_ids: set,
                           players_df: pd.DataFrame | None = None) -> tuple[set, dict]:
    """
    Cruza el PBP con rookie_ids (GSIS IDs) para obtener los nombres
    tal como aparecen en el PBP — evita falsos positivos por abreviaturas.
    """
    ID_NAME_PAIRS = [
        ("passer_player_id",               "passer_player_name"),
        ("receiver_player_id",             "receiver_player_name"),
        ("rusher_player_id",               "rusher_player_name"),
        ("sack_player_id",                 "sack_player_name"),
        ("forced_fumble_player_1_player_id", "forced_fumble_player_1_player_name"),
        ("forced_fumble_player_2_player_id", "forced_fumble_player_2_player_name"),
        ("pass_defense_1_player_id",       "pass_defense_1_player_name"),
        ("pass_defense_2_player_id",       "pass_defense_2_player_name"),
        ("tackle_for_loss_1_player_id",    "tackle_for_loss_1_player_name"),
        ("tackle_for_loss_2_player_id",    "tackle_for_loss_2_player_name"),
        ("interception_player_id",         "interception_player_name"),
    ]
    # id → full name desde players.csv
    # Solo columnas que tienen nombre completo (first + last); excluimos football_name (puede ser solo el primero)
    id_to_full = {}
    if players_df is not None:
        gsis_col = next((c for c in ["gsis_id"] if c in players_df.columns), None)
        full_col  = next((c for c in ["display_name", "full_name"]
                          if c in players_df.columns), None)
        if gsis_col and full_col:
            rooks_pl = players_df[players_df[gsis_col].astype(str).isin(rookie_ids)]
            for _, row in rooks_pl.iterrows():
                gsis = str(row[gsis_col])
                name = str(row[full_col])
                # solo aceptamos si tiene al menos un espacio (nombre + apellido) y no es nan
                if gsis not in id_to_full and " " in name and name.lower() != "nan":
                    id_to_full[gsis] = name

    pbp_rookie_names = set()
    norm_to_full: dict[str, str] = {}   # "jwilliams" → "Jameson Williams"

    for id_col, name_col in ID_NAME_PAIRS:
        if id_col in pbp_df.columns and name_col in pbp_df.columns:
            sub = pbp_df[[id_col, name_col]].dropna(subset=[id_col, name_col])
            sub = sub[sub[id_col].astype(str).isin(rookie_ids)]
            for _, row in sub.drop_duplicates(subset=[id_col]).iterrows():
                norm = _norm_name(str(row[name_col]))
                pbp_rookie_names.add(norm)
                if norm not in norm_to_full:
                    full = id_to_full.get(str(row[id_col]), "")
                    if full:
                        norm_to_full[norm] = full

    return pbp_rookie_names, norm_to_full


def _is_rook_by_id(series, id_col, rookie_ids):
    """Devuelve máscara booleana: True si el player_id de la fila está en rookie_ids."""
    if id_col and id_col in series.columns:
        return series[id_col].astype(str).isin(rookie_ids)
    return pd.Series(False, index=series.index)


# ---------------- Ataque (vectorizado) ----------------
def calc_ataque(d, play_type, passer, receiver, rusher, posteam, rookie_ids,
                passer_id=None, receiver_id=None, rusher_id=None):
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

            # rookies QB — por ID
            passes["_is_rook"] = _is_rook_by_id(passes, passer_id, rookie_ids)
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

                    # rookies receptor — por ID
                    rec_plays["_is_rook"] = _is_rook_by_id(rec_plays, receiver_id, rookie_ids)
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

            # rookies rusher — por ID
            runs["_is_rook"] = _is_rook_by_id(runs, rusher_id, rookie_ids)
            rook_runs = runs[runs["_is_rook"]]
            if not rook_runs.empty:
                rk_run = rook_runs.groupby("_key")["epa"].sum()
                for k, v in rk_run.items():
                    rook_credits[k] = rook_credits.get(k, 0.0) + v

    return credits, rook_credits


# ---------------- Defensa (vectorizado) ----------------
def _add_def_credits(sub, name_col1, name_col2, team_col, credits, rook_credits,
                     rookie_ids, id_col1=None, id_col2=None, multiplier=-1.0):
    """Para jugadas defensivas con hasta 2 jugadores, acumula crédito dividido."""
    if sub.empty:
        return

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

    def _accum(rows, col, id_col, factor):
        if rows.empty or not col:
            return
        rows = rows.copy()
        rows["_key"] = make_key(rows[col], rows[team_col] if team_col else pd.Series("?", index=rows.index))
        totals = rows.groupby("_key")["epa"].sum() * multiplier * factor
        for k, v in totals.items():
            credits[k] = credits.get(k, 0.0) + v
        # rookies — por ID
        rows["_is_rook"] = _is_rook_by_id(rows, id_col, rookie_ids)
        rook_rows = rows[rows["_is_rook"]]
        if not rook_rows.empty:
            rk = rook_rows.groupby("_key")["epa"].sum() * multiplier * factor
            for k, v in rk.items():
                rook_credits[k] = rook_credits.get(k, 0.0) + v

    _accum(only1, name_col1, id_col1, 1.0)
    _accum(only2, name_col2, id_col2, 1.0)
    _accum(both,  name_col1, id_col1, 0.5)
    _accum(both,  name_col2, id_col2, 0.5)


def calc_defensa(d, int_nm, sack_nm, tfl1_nm, tfl2_nm, ff1_nm, ff2_nm, pd1_nm, pd2_nm, defteam, rookie_ids,
                 int_id=None, sack_id=None, tfl1_id=None, tfl2_id=None,
                 ff1_id=None, ff2_id=None, pd1_id=None, pd2_id=None):
    credits = {}
    rook_credits = {}

    base = d[d["epa"].notna()]

    # Intercepciones
    if int_nm:
        sub = base[base[int_nm].notna()]
        _add_def_credits(sub, int_nm, None, defteam, credits, rook_credits, rookie_ids, id_col1=int_id)

    # Sacks
    if sack_nm and "sack" in d.columns:
        sub = base[(base["sack"] == 1) & base[sack_nm].notna()]
        _add_def_credits(sub, sack_nm, None, defteam, credits, rook_credits, rookie_ids, id_col1=sack_id)

    # TFL
    if tfl1_nm or tfl2_nm:
        mask = pd.Series(False, index=base.index)
        if tfl1_nm: mask |= base[tfl1_nm].notna()
        if tfl2_nm: mask |= base[tfl2_nm].notna()
        _add_def_credits(base[mask], tfl1_nm, tfl2_nm, defteam, credits, rook_credits, rookie_ids, id_col1=tfl1_id, id_col2=tfl2_id)

    # Fumbles forzados
    if ff1_nm or ff2_nm:
        mask = pd.Series(False, index=base.index)
        if ff1_nm: mask |= base[ff1_nm].notna()
        if ff2_nm: mask |= base[ff2_nm].notna()
        _add_def_credits(base[mask], ff1_nm, ff2_nm, defteam, credits, rook_credits, rookie_ids, id_col1=ff1_id, id_col2=ff2_id)

    # Passes defendidos
    if pd1_nm or pd2_nm:
        mask = pd.Series(False, index=base.index)
        if pd1_nm: mask |= base[pd1_nm].notna()
        if pd2_nm: mask |= base[pd2_nm].notna()
        _add_def_credits(base[mask], pd1_nm, pd2_nm, defteam, credits, rook_credits, rookie_ids, id_col1=pd1_id, id_col2=pd2_id)

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
    global SEASON
    df, SEASON = cargar_pbp(SEASON)
    print(f"PBP {SEASON}: {len(df):,} jugadas REG")

    play_type = pick_col(df, "play_type")
    if play_type is None or "epa" not in df.columns:
        raise SystemExit("Faltan columnas esenciales (play_type o epa).")

    to_num(df, ["epa", "sack"])
    posteam = pick_col(df, "posteam")
    defteam = pick_col(df, "defteam")

    # Nombre columns
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

    # ID columns (para detección de rookies sin ambigüedad)
    passer_id_nm   = pick_col(df, "passer_player_id")
    receiver_id_nm = pick_col(df, "receiver_player_id")
    rusher_id_nm   = pick_col(df, "rusher_player_id")
    int_id_nm   = pick_col(df, "interception_player_id")
    sack_id_nm  = pick_col(df, "sack_player_id")
    tfl1_id_nm  = pick_col(df, "tackle_for_loss_1_player_id")
    tfl2_id_nm  = pick_col(df, "tackle_for_loss_2_player_id")
    ff1_id_nm   = pick_col(df, "forced_fumble_player_1_player_id")
    ff2_id_nm   = pick_col(df, "forced_fumble_player_2_player_id")
    pd1_id_nm   = pick_col(df, "pass_defense_1_player_id")
    pd2_id_nm   = pick_col(df, "pass_defense_2_player_id")

    # Rookies: IDs desde players.csv
    print("Descargando players.csv para detectar rookies...")
    players = pd.read_csv(URL_PLAYERS, low_memory=False)
    rookie_ids, _ = build_rookie_sets(players)
    print(f"  Rookies por ID (players.csv): {len(rookie_ids)}")
    _, rookie_fullnames = build_pbp_rookie_names(df, rookie_ids, players)
    print(f"  Rookies con jugadas en PBP: {len(rookie_fullnames)}")

    # Calcular
    of_credit, of_rook   = calc_ataque(
        df, play_type, passer_nm, receiver_nm, rusher_nm, posteam, rookie_ids,
        passer_id=passer_id_nm, receiver_id=receiver_id_nm, rusher_id=rusher_id_nm)
    def_credit, def_rook = calc_defensa(
        df, int_nm, sack_nm, tfl1_nm, tfl2_nm, ff1_nm, ff2_nm, pd1_nm, pd2_nm, defteam, rookie_ids,
        int_id=int_id_nm, sack_id=sack_id_nm, tfl1_id=tfl1_id_nm, tfl2_id=tfl2_id_nm,
        ff1_id=ff1_id_nm, ff2_id=ff2_id_nm, pd1_id=pd1_id_nm, pd2_id=pd2_id_nm)
    st_credit = calc_st(df, play_type, kr_nm, pr_nm, kicker_nm, punter_nm, posteam)

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

    def fmt(name, val, is_rookie=False):
        if name is None:
            return "sin datos"
        s = f"{name}  EPA {val:+.3f}"
        if is_rookie and name:
            # name es "X.Apellido (TEAM)" → extraemos la parte antes del espacio
            abbrev = name.split("(")[0].strip()
            full = rookie_fullnames.get(_norm_name(abbrev), "")
            if full:
                s += f"  [{full}]"
        return s

    print(f"\n========== LIDERES EPA — TEMPORADA {SEASON} ==========")
    print(f"  ATAQUE             : {fmt(of_name,   of_val)}")
    print(f"  DEFENSA            : {fmt(def_name,  def_val)}")
    print(f"  ROOKIE ATAQUE      : {fmt(of_r_name, of_r_val,  is_rookie=True)}")
    print(f"  ROOKIE DEFENSA     : {fmt(dr_name,   dr_val,    is_rookie=True)}")
    print(f"  EQUIPOS ESPECIALES : {fmt(st_name,   st_val)}")

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
