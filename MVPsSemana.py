# MVPsSemana.py
# Líderes semanales de EPA — Ataque, Defensa, Equipos Especiales y Rookie
# Ataque: 50% QB / 50% receptor en pases, 100% portador en carreras
# Defensa: EPA negativo generado por intercepciones y sacks
# Equipos especiales: retornos, FGs, XPs y punts

import pandas as pd
import numpy as np
from pbp_loader import cargar_pbp, week_cli

# === Config ===
SEASON = None   # None = auto-detectar última temporada

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


def _serie(d, name):
    """La columna si existe; si no, una de ceros (no todos los años la traen)."""
    return d[name] if name in d.columns else pd.Series(0, index=d.index)


def descartes_retorno(d, ret_col):
    """{motivo: máscara} de las jugadas que NO se acreditan al retornador.

    El EPA de la jugada entera no es suyo: hay que quitar lo que no hizo él.
    No se reparte EPA dentro de la jugada (haría falta un modelo de EP propio),
    se descarta la jugada completa cuando el crédito no es atribuible.
    """
    sin_retorno = (
        (_serie(d, "punt_fair_catch").fillna(0) == 1)
        | (_serie(d, "kickoff_fair_catch").fillna(0) == 1)
        | (_serie(d, "touchback").fillna(0) == 1)
        | (_serie(d, "punt_downed").fillna(0) == 1)
        | (_serie(d, "kickoff_downed").fillna(0) == 1)
        | (_serie(d, "punt_out_of_bounds").fillna(0) == 1)
        | (_serie(d, "kickoff_out_of_bounds").fillna(0) == 1)
        | (d["return_yards"].isna() if "return_yards" in d.columns else False)
    )

    # Fumble del retornador que recupera su propio equipo: la ganancia
    # posterior es del compañero. Si lo PIERDE, el EPA es suyo y se queda.
    ret = d[ret_col].fillna("").str.strip()
    fumblo = pd.Series(False, index=d.index)
    for c in ("fumbled_1_player_name", "fumbled_2_player_name"):
        if c in d.columns:
            fumblo |= (d[c].fillna("").str.strip() == ret) & (ret != "")
    fumble_propio = fumblo & (_serie(d, "fumble_lost").fillna(0) != 1)

    # Solo las penalties APLICADAS mueven el EPA: las declinadas y las
    # compensadas vienen con penalty==1 y penalty_yards 0, y la jugada vale
    # tal cual.
    con_penalty = (_serie(d, "penalty").fillna(0) == 1) & (
        _serie(d, "penalty_yards").fillna(0) > 0
    )
    # Salvo el TD del propio retornador: si la penalty hubiera anulado el
    # retorno no habria TD, asi que es posterior a la jugada (antideportiva
    # tras anotar, cobrada en el kickoff siguiente). El 29-sep-2026 esto
    # dejaba fuera el punt return de 86 yds de M.Price (MIN) en la semana 3
    td_propio = (_serie(d, "touchdown").fillna(0) == 1) & (
        d["td_player_name"].fillna("").str.strip() == ret
        if "td_player_name" in d.columns else False
    ) & (ret != "")
    con_penalty = con_penalty & ~td_propio

    return {
        "sin retorno (fair catch, touchback, downed, fuera)": sin_retorno,
        "fumble del retornador recuperado por su equipo": fumble_propio & ~sin_retorno,
        "penalty aplicada en la jugada": con_penalty & ~sin_retorno & ~fumble_propio,
    }


def calc_st(d, play_type, kr_ret, pr_ret, kicker, punter, posteam, defteam,
            traza=None):
    """EPA de equipos especiales, acreditado a quien lo genera.

    Hasta sep-2026 esto daba al retornador el EPA de la JUGADA ENTERA, y se
    le colgaban tres cosas que no hizo: el tramo posterior a un fumble suyo
    que recupera un compañero, las penalties del equipo que patea y el EPA
    (cambiado de signo) de punts que solo hizo fair catch —que mide al
    PATEADOR rival—. Con eso R.Shaheed salía líder de la semana 2 de 2026 con
    +6,731 cuando lo suyo eran +0,924. Ver docs/decisiones.md (22-sep-2026).

    Kickers y punters no se tocan: el EPA de un FG o un XP es suyo entero.
    """
    credits = {}

    def _add(plays, name_col, team_col, multiplier=1.0):
        if name_col and not plays.empty:
            sub = plays[plays[name_col].notna()].copy()
            if not sub.empty:
                sub["_key"] = make_key(sub[name_col], sub[team_col] if team_col else pd.Series("", index=sub.index))
                totals = sub.groupby("_key")["epa"].sum() * multiplier
                for k, v in totals.items():
                    credits[k] = credits.get(k, 0.0) + v

    if not play_type:
        return credits

    base = d[d["epa"].notna()]

    # Los retornos no son play_type propio en nflverse: van dentro de
    # "kickoff" (posteam = equipo que recibe) y "punt" (posteam = equipo
    # que patea → el retornador es del defteam y su EPA bueno es negativo).
    for tipo, ret_col, team_col, mult in [("kickoff", kr_ret, posteam,  1.0),
                                          ("punt",    pr_ret, defteam, -1.0)]:
        if not ret_col:
            continue
        jug = base[(base[play_type] == tipo) & base[ret_col].notna()]
        if jug.empty:
            continue
        fuera = pd.Series(False, index=jug.index)
        for motivo, mask in descartes_retorno(jug, ret_col).items():
            fuera |= mask
            if traza is not None and mask.any():
                traza.append((tipo, motivo, int(mask.sum()),
                              round(float(jug.loc[mask, "epa"].sum() * mult), 2)))
        _add(jug[~fuera], ret_col, team_col, multiplier=mult)

    _add(base[base[play_type] == "field_goal"],  kicker, posteam)
    _add(base[base[play_type] == "extra_point"], kicker, posteam)
    # Un punt que el retornador rival suelta y recupera el equipo que patea
    # (muff) vale un monton de EPA que no es del punter: es el error del
    # retornador y el merito de quien cae encima. Semana 3 de 2026: Gillikin
    # (ARI) +6.670, de los que +5.96 eran un muff de S.Neal (SF)
    punts = base[base[play_type] == "punt"]
    muff = (_serie(punts, "fumble_lost").fillna(0) == 1)
    _add(punts[~muff],                           punter, posteam)

    return credits


def claves_rookies(d, season):
    """Claves 'Nombre (TEAM)' de los rookies que aparecen en la semana.

    El cruce va por gsis_id y no por nombre: el PBP abrevia ('J.Trotter') y
    en 2026 hay dos J.Trotter en la liga, uno rookie (TB) y otro no (PHI).
    """
    from pbp_loader import cargar_rosters, DatosNoDisponibles
    try:
        ros, _ = cargar_rosters(season)
    except DatosNoDisponibles as e:
        print(f"  (sin roster {season}: no se puede sacar el MVP rookie — {e})")
        return set()
    ids = set(ros.loc[ros["rookie_year"] == season, "gsis_id"].dropna())
    roles = [("passer_player_name", "passer_player_id", "posteam"),
             ("receiver_player_name", "receiver_player_id", "posteam"),
             ("rusher_player_name", "rusher_player_id", "posteam"),
             ("interception_player_name", "interception_player_id", "defteam"),
             ("sack_player_name", "sack_player_id", "defteam"),
             ("kicker_player_name", "kicker_player_id", "posteam"),
             ("punter_player_name", "punter_player_id", "posteam"),
             ("kickoff_returner_player_name", "kickoff_returner_player_id", "posteam"),
             ("punt_returner_player_name", "punt_returner_player_id", "defteam")]
    claves = set()
    for nom, pid, team in roles:
        if nom in d.columns and pid in d.columns:
            sub = d[d[pid].isin(ids) & d[nom].notna()]
            claves |= set(make_key(sub[nom], sub[team]))
    return claves


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
    semana_str = str(week_cli() or "") or input("Semana (numero): ").strip()
    try:
        week = int(semana_str)
    except ValueError:
        raise SystemExit("Semana invalida.")

    global SEASON
    # solo_reg=False: el filtro semanal es del usuario (semanas 19+ = playoffs)
    df, SEASON = cargar_pbp(SEASON, solo_reg=False)
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
    st_credit  = calc_st(d, play_type, kr_ret, pr_ret, kicker, punter, posteam, defteam)

    print(f"\n========== LIDERES EPA — Semana {week} NFL {SEASON} ==========")
    of_series  = print_top("ATAQUE",             of_credit)
    def_series = print_top("DEFENSA",            def_credit)
    st_series  = print_top("EQUIPOS ESPECIALES", st_credit)

    # ROOKIE: su EPA sumado en las tres fases (un rookie casi nunca puntua en
    # dos, pero si pasa, cuenta todo lo que genero)
    rookies = claves_rookies(d, SEASON)
    rk_credit = {}
    for cred in (of_credit, def_credit, st_credit):
        for k, v in cred.items():
            if k in rookies:
                rk_credit[k] = rk_credit.get(k, 0.0) + v
    rk_series = print_top("ROOKIE", rk_credit)

    show_top3 = input("\nMostrar top-3 por categoria? (s/n): ").strip().lower()
    if show_top3 == "s":
        for label, series in [("ATAQUE", of_series), ("DEFENSA", def_series),
                              ("ST", st_series), ("ROOKIE", rk_series)]:
            if series is not None:
                print(f"\nTop-3 {label}:")
                print(series.head(3).apply(lambda v: f"{v:+.3f}").to_string())

if __name__ == "__main__":
    main()
