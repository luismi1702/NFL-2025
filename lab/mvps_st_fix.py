# lab/mvps_st_fix.py
# Arreglo del credito de EQUIPOS ESPECIALES de MVPsSemana.calc_st, comparado
# contra el actual en las semanas que se le pasen.
#
# El problema (detectado en la sem. 2 de 2026 con R.Shaheed, +6.731):
# calc_st acredita al retornador el EPA de la JUGADA ENTERA, asi que se le
# cuelgan tres cosas que no hizo el:
#   (a) el tramo posterior a un fumble suyo que recupera un companero,
#   (b) las penalties del equipo que patea,
#   (c) el EPA (cambiado de signo) de punts que solo hizo fair catch, que
#       mide al PATEADOR rival y no a el.
#
# calc_st_v2 no reparte el EPA dentro de la jugada (para eso haria falta un
# modelo de EP propio): descarta las jugadas en las que el credito no es
# atribuible y deja el resto igual. Kickers y punters no se tocan.
#
#   python lab/mvps_st_fix.py --season 2026            -> semanas 1 y 2
#   python lab/mvps_st_fix.py --season 2026 --week 2   -> solo la 2

import sys, os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pbp_loader import cargar_pbp, season_cli, week_cli
from MVPsSemana import calc_st, pick_col, make_key, to_num

SEMANAS_POR_DEFECTO = [1, 2]


def _col(d, name):
    return d[name] if name in d.columns else pd.Series(0, index=d.index)


def _fumblo_el_retornador(d, ret_col):
    """True en las jugadas donde el fumble lo comete el propio retornador."""
    ret = d[ret_col].fillna("").str.strip()
    fumblo = pd.Series(False, index=d.index)
    for c in ("fumbled_1_player_name", "fumbled_2_player_name"):
        if c in d.columns:
            fumblo |= (d[c].fillna("").str.strip() == ret) & (ret != "")
    return fumblo


def motivos_descarte(d, ret_col, es_punt):
    """Devuelve {motivo: mascara} de las jugadas que NO se acreditan."""
    sin_retorno = (
        (_col(d, "punt_fair_catch").fillna(0) == 1)
        | (_col(d, "kickoff_fair_catch").fillna(0) == 1)
        | (_col(d, "touchback").fillna(0) == 1)
        | (_col(d, "punt_downed").fillna(0) == 1)
        | (_col(d, "kickoff_downed").fillna(0) == 1)
        | (_col(d, "punt_out_of_bounds").fillna(0) == 1)
        | (_col(d, "kickoff_out_of_bounds").fillna(0) == 1)
        | (d["return_yards"].isna() if "return_yards" in d.columns else False)
    )

    # Fumble del retornador que recupera su propio equipo: la ganancia
    # posterior es del companero. Si lo PIERDE, el EPA es suyo y se queda.
    fumblo = _fumblo_el_retornador(d, ret_col)
    perdido = _col(d, "fumble_lost").fillna(0) == 1
    fumble_propio = fumblo & ~perdido

    # Solo las penalties APLICADAS mueven el EPA: las declinadas y las
    # compensadas vienen con penalty==1 y penalty_yards 0, y la jugada vale
    # tal cual (8 de las 54 de las semanas 1-2 de 2026).
    con_penalty = (_col(d, "penalty").fillna(0) == 1) & (
        _col(d, "penalty_yards").fillna(0) > 0
    )

    return {
        "sin retorno (fair catch, touchback, downed, fuera)": sin_retorno,
        "fumble del retornador recuperado por su equipo": fumble_propio & ~sin_retorno,
        "penalty aplicada en la jugada": con_penalty & ~sin_retorno & ~fumble_propio,
    }


def calc_st_v2(d, play_type, kr_ret, pr_ret, kicker, punter, posteam, defteam,
               traza=None):
    credits = {}

    def _add(plays, name_col, team_col, multiplier=1.0):
        if name_col and not plays.empty:
            sub = plays[plays[name_col].notna()].copy()
            if not sub.empty:
                sub["_key"] = make_key(sub[name_col], sub[team_col])
                totals = sub.groupby("_key")["epa"].sum() * multiplier
                for k, v in totals.items():
                    credits[k] = credits.get(k, 0.0) + v

    if not play_type:
        return credits
    base = d[d["epa"].notna()]

    for tipo, ret_col, team_col, mult in [
        ("kickoff", kr_ret, posteam, 1.0),
        ("punt", pr_ret, defteam, -1.0),
    ]:
        jug = base[(base[play_type] == tipo) & base[ret_col].notna()]
        if jug.empty:
            continue
        motivos = motivos_descarte(jug, ret_col, es_punt=(tipo == "punt"))
        fuera = pd.Series(False, index=jug.index)
        for motivo, mask in motivos.items():
            fuera |= mask
            if traza is not None and mask.any():
                traza.append((tipo, motivo, int(mask.sum()),
                              round(float(jug.loc[mask, "epa"].sum() * mult), 2)))
        _add(jug[~fuera], ret_col, team_col, multiplier=mult)

    # Kickers y punters: sin cambios. El EPA de un FG o un XP es suyo entero.
    _add(base[base[play_type] == "field_goal"], kicker, posteam)
    _add(base[base[play_type] == "extra_point"], kicker, posteam)
    _add(base[base[play_type] == "punt"], punter, posteam)
    return credits


def top(credits, n=5):
    return pd.Series(credits).sort_values(ascending=False).head(n)


def main():
    season = season_cli()
    df, season = cargar_pbp(season, solo_reg=False)
    to_num(df, ["week", "epa"])

    w = week_cli()
    semanas = [int(w)] if w else SEMANAS_POR_DEFECTO

    for week in semanas:
        d = df[df["week"] == week].copy()
        if d.empty:
            print(f"\n### Semana {week}: sin jugadas")
            continue

        args = dict(
            play_type=pick_col(d, "play_type"),
            kr_ret=pick_col(d, "kickoff_returner_player_name"),
            pr_ret=pick_col(d, "punt_returner_player_name"),
            kicker=pick_col(d, "kicker_player_name", "kicker"),
            punter=pick_col(d, "punter_player_name", "punter"),
            posteam=pick_col(d, "posteam"),
            defteam=pick_col(d, "defteam"),
        )

        traza = []
        viejo = calc_st(d, **args)
        nuevo = calc_st_v2(d, traza=traza, **args)

        print(f"\n{'='*66}\n  SEMANA {week} — NFL {season}\n{'='*66}")
        print("\n  ACTUAL (calc_st)              ARREGLADO (calc_st_v2)")
        v, n = top(viejo), top(nuevo)
        for i in range(5):
            izq = f"{v.index[i]:<22.22} {v.iloc[i]:+7.3f}" if i < len(v) else ""
            der = f"{n.index[i]:<22.22} {n.iloc[i]:+7.3f}" if i < len(n) else ""
            print(f"  {izq}      {der}")

        lider_v, lider_n = v.index[0], n.index[0]
        print(f"\n  Lider: {lider_v} -> {lider_n}"
              + ("   (CAMBIA)" if lider_v != lider_n else "   (mismo)"))

        print("\n  Jugadas descartadas del credito a retornadores:")
        for tipo, motivo, cuantas, epa in traza:
            print(f"    {tipo:8s} {motivo:<50.50} {cuantas:3d} jug.  {epa:+7.2f} EPA")

        print("\n  Mayores caidas:")
        dif = (pd.Series(nuevo) - pd.Series(viejo)).dropna().sort_values()
        for k, val in dif.head(5).items():
            print(f"    {k:<24.24} {viejo[k]:+7.3f} -> {nuevo[k]:+7.3f}  ({val:+.3f})")


if __name__ == "__main__":
    main()
