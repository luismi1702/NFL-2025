"""
tendencias_historicas.py — las tendencias de NFL Next Gen Trends, por equipo y
temporada, calculadas con NUESTROS datos públicos (13-nov-2026). Sirve para la
historia que la web solo da por su API: 2018-2025 con participación (FTN/NGS)
y PBP de nflverse, con el EPA de nflverse (el nuestro, no el de NGS).

Las definiciones imitan las de Next Gen Trends y se comprueban contra sus
medias de la liga (datos_web/ngs_trends_liga_*.csv): el script imprime la
diferencia por métrica y temporada. Lo que no cuadre dentro de ±3 puntos se
avisa: esa métrica NO es comparable con la de la web.

  Ataque: 11/12/13/21 personal, pase pedido, quick passing (<2,5 s), pases de
          10+ yardas de aire, rutas hacia dentro y hacia fuera del objetivo,
          motion, play action en early downs y sin huddle (estos tres, FTN:
          solo 2022+)
  Defensa: base/nickel/dime, man/zone, single-high/split-safety, Cover 0,
          blitz (5+ rushers) y presión con y sin blitz
  Sin datos públicos: conceptos de bloqueo (zone, gap, man) y press.

Salida: pbp_cache/tendencias_equipo_hist.parquet (y .csv), con las mismas
columnas que datos_web/ngs_trends_equipos_* (equipo, lado, metrica, temporada,
jugadas, elegibles, uso, puesto, epa, exito, big_play, uso_liga).

    python tendencias_historicas.py              # 2018-2025
    python tendencias_historicas.py --desde 2022
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

from pbp_loader import cargar_ftn, cargar_participation, cargar_pbp

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(RAIZ, "pbp_cache", "tendencias_equipo_hist")

DENTRO = {"SLANT", "IN", "IN/DIG", "POST", "CROSS", "SHALLOW CROSS/DRAG", "TEXAS/ANGLE", "ANGLE"}
FUERA = {"OUT", "QUICK OUT", "DEEP OUT", "CORNER", "FLAT"}  # SWING y WHEEL no: así cuadra con NGS
UN_ALTO = {"COVER_1", "COVER_3"}
DOS_ALTOS = {"COVER_2", "2_MAN", "COVER_4", "COVER_6", "COVER_9"}


def contar(texto, posiciones):
    if not isinstance(texto, str):
        return np.nan
    return sum(int(n) for n, pos in re.findall(r"(\d+) ([A-Z]+)", texto) if pos in posiciones)


def personal_ataque(texto):
    rb, te = contar(texto, {"RB", "FB"}), contar(texto, {"TE"})
    return np.nan if np.isnan(rb) else f"{int(rb)}{int(te)}"


def defensas(texto):
    return contar(texto, {"CB", "S", "FS", "SS", "DB"})


def jugadas(season):
    pbp, _ = cargar_pbp(season)
    pbp = pbp[pbp.play_type.isin(["pass", "run"]) & (pbp.qb_kneel == 0) & (pbp.qb_spike == 0)
              & pbp.posteam.notna() & pbp.epa.notna()].copy()
    pbp["big"] = np.where(pbp["pass"] == 1, pbp.yards_gained >= 20, pbp.yards_gained >= 10)
    part, _ = cargar_participation(season)
    part = part.rename(columns={"nflverse_game_id": "game_id"})
    cols = ["game_id", "play_id", "offense_personnel", "defense_personnel", "number_of_pass_rushers",
            "time_to_throw", "was_pressure", "route", "defense_man_zone_type", "defense_coverage_type"]
    d = pbp.merge(part[cols], on=["game_id", "play_id"], how="left")
    d["pers"] = d.offense_personnel.map(personal_ataque)
    d["n_db"] = d.defense_personnel.map(defensas)
    if season >= 2022:
        try:
            ftn, _ = cargar_ftn(season)
            ftn = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
            d = d.merge(ftn[["game_id", "play_id", "is_motion", "is_play_action", "is_no_huddle"]],
                        on=["game_id", "play_id"], how="left")
        except Exception as e:  # noqa: BLE001
            print(f"  {season}: sin FTN ({type(e).__name__})")
    return d


def definiciones(d):
    """(lado, métrica, máscara de elegibles, máscara de 'sí')."""
    ataque = pd.Series(True, index=d.index)
    drop = d["pass"] == 1
    tiro = (d.pass_attempt == 1) & (d.sack == 0)
    early = d.down.isin([1, 2])
    rec = d.route.fillna("").str.upper()
    con_ruta = tiro & (rec != "")
    cob = d.defense_coverage_type.fillna("")
    con_cob = drop & (cob != "") & ~cob.isin(["BLOWN", "PREVENT", "COMBO"])
    mz = d.defense_man_zone_type.fillna("")
    con_mz = drop & mz.isin(["MAN_COVERAGE", "ZONE_COVERAGE"])
    con_pers = d.pers.notna()
    con_db = d.n_db.notna() & (d.n_db > 0)
    rushers = d.number_of_pass_rushers
    con_rush = drop & rushers.notna()
    blitz = rushers >= 5
    con_pres = con_rush & d.was_pressure.notna()
    out = []
    for g in ["11", "12", "13", "21"]:
        out.append(("ataque", f"{g}Personnel", con_pers, d.pers == g))
    out += [("ataque", "calledPassRate", ataque, drop),
            ("ataque", "quickPassing", tiro & d.time_to_throw.notna(), d.time_to_throw < 2.5),
            ("ataque", "downfieldPassing", tiro & d.air_yards.notna(), d.air_yards >= 10),
            ("ataque", "inBreakingTargets", con_ruta, rec.isin(DENTRO)),
            ("ataque", "outBreakingTargets", con_ruta, rec.isin(FUERA))]
    if "is_motion" in d:
        ftn = d.is_motion.notna()
        out += [("ataque", "motion", ftn, d.is_motion == 1),
                ("ataque", "earlyDownPlayAction", ftn & drop & early, d.is_play_action == 1),
                ("ataque", "noHuddle", ftn, d.is_no_huddle == 1)]
    out += [("defensa", "base", con_db, d.n_db <= 4),
            ("defensa", "nickel", con_db, d.n_db == 5),
            ("defensa", "dime", con_db, d.n_db >= 6),
            ("defensa", "manCoverage", con_mz, mz == "MAN_COVERAGE"),
            ("defensa", "zoneCoverage", con_mz, mz == "ZONE_COVERAGE"),
            ("defensa", "singleHigh", con_cob, cob.isin(UN_ALTO)),
            ("defensa", "splitSafety", con_cob, cob.isin(DOS_ALTOS)),
            ("defensa", "cover0", con_cob, cob == "COVER_0"),
            ("defensa", "blitzRate", con_rush, blitz),
            ("defensa", "pressureRateBlitz", con_pres & blitz, d.was_pressure == 1),
            ("defensa", "pressureRateNoBlitz", con_pres & ~blitz, d.was_pressure == 1)]
    return out


def tabla(d, season):
    filas = []
    for lado, met, eleg, si in definiciones(d):
        eleg, si = eleg.fillna(False).astype(bool), si.fillna(False).astype(bool)
        equipo = d.posteam if lado == "ataque" else d.defteam
        base = d[eleg].assign(eq=equipo[eleg], si=si[eleg])
        if base.empty:
            continue
        uso_liga = base.si.mean()
        # presión: el "uso" es el % de presión; EPA y éxito, de esas mismas jugadas
        g = base.groupby("eq")
        res = pd.DataFrame({"elegibles": g.size(), "jugadas": g.si.sum(), "uso": g.si.mean()})
        sub = base[base.si].groupby("eq")
        res["epa"] = sub.epa.mean()
        res["exito"] = sub.success.mean()
        res["big_play"] = sub.big.mean()
        alto_mejor = True  # puesto 1 = más uso, como en la web
        res["puesto"] = res.uso.rank(ascending=not alto_mejor, method="min").astype(int)
        res["uso_liga"] = uso_liga
        res = res.reset_index().rename(columns={"eq": "equipo"})
        res["lado"], res["metrica"], res["temporada"] = lado, met, season
        filas.append(res)
    return pd.concat(filas, ignore_index=True)


def contrastar(df):
    """Nuestra media de la liga frente a la de Next Gen Trends."""
    rutas = sorted(glob.glob(os.path.join(RAIZ, "datos_web", "ngs_trends_liga_*.csv")))
    if not rutas:
        print("Sin datos_web/ngs_trends_liga_*.csv: lanza copiar_ngs_trends.py para contrastar")
        return
    ngs = pd.read_csv(rutas[-1]).rename(columns={"uso_liga": "ngs"})
    nos = df.groupby(["metrica", "temporada"]).uso_liga.first().reset_index()
    c = nos.merge(ngs[["metrica", "temporada", "ngs"]], on=["metrica", "temporada"])
    c["dif"] = (c.uso_liga - c.ngs) * 100
    piv = c.pivot(index="metrica", columns="temporada", values="dif").round(1)
    print("\nDiferencia con Next Gen Trends (puntos de %, nuestra media - la suya):")
    print(piv.to_string())
    malas = piv.abs().max(axis=1)
    malas = malas[malas > 3]
    if len(malas):
        print("\n!! Alguna temporada difiere más de 3 puntos de la web: " + ", ".join(malas.index))
    return c[["metrica", "temporada", "ngs", "dif"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", type=int, default=2018)
    ap.add_argument("--hasta", type=int, default=2025)
    a = ap.parse_args()
    partes = []
    for s in range(a.desde, a.hasta + 1):
        d = jugadas(s)
        t = tabla(d, s)
        print(f"{s}: {len(d)} jugadas, {t.metrica.nunique()} métricas")
        partes.append(t)
    df = pd.concat(partes, ignore_index=True)
    c = contrastar(df)
    if c is not None:
        df = df.merge(c.rename(columns={"ngs": "uso_liga_ngs", "dif": "dif_ngs"}),
                      on=["metrica", "temporada"], how="left")
        # comparable con la web si la media de la liga cuadra dentro de 3 puntos
        df["comparable_ngs"] = df.dif_ngs.abs() <= 3
    df["fuente"] = "PBP nflverse + participación (FTN/NGS) + FTN charting; EPA de nflverse"
    df.to_parquet(SALIDA + ".parquet", index=False)
    df.to_csv(SALIDA + ".csv", index=False, encoding="utf-8")
    print(f"-> {SALIDA}.parquet ({len(df)} filas)")


if __name__ == "__main__":
    main()
