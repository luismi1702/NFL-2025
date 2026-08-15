# -*- coding: utf-8 -*-
"""
lab/cirugia_radares.py — P3 auditoría visual (jul-2026).
Sustituye la 6ª métrica "EPA on/off" (N/D para casi todos los titulares con
MIN_SNAPS=100) por producción por partido, y elimina la carga de
pbp_participation (lenta) de los 5 comparadores defensivos.
"""
import re


def leer(f):
    return open(f, encoding="utf-8").read()


def escribir(f, s):
    open(f, "w", encoding="utf-8").write(s)


def borrar_seccion(src, inicio, fin, etiqueta):
    m1 = re.search(inicio, src, re.M)
    m2 = re.search(fin, src, re.M)
    assert m1 and m2 and m1.start() < m2.start(), f"anclas no encontradas: {etiqueta}"
    return src[:m1.start()] + src[m2.start():]


def reemplazar(src, viejo, nuevo, etiqueta):
    assert viejo in src, f"no encontrado: {etiqueta}"
    return src.replace(viejo, nuevo, 1)


BLOQUE_ONOFF_CBS = '''    gsis_id    = get_gsis_id(stats_row)
    team       = get_player_team(stats_row)
    epa_on_off = get_on_off_epa(gsis_id, team, pbp_df, part_df)

    ff = float(stats_row.get("def_fumbles_forced", float("nan")))

    return {
        "Pases defendidos":  pds,
        "Intercepciones":    ints,
        "Tackles":           tackles,
        "TFL":               tfl,
        "Fumbles forzados":  ff,
        "EPA on/off":        epa_on_off,
    }'''
NUEVO_CBS = '''    ff = float(stats_row.get("def_fumbles_forced", float("nan")))

    games = float(stats_row.get("games", 0) or 0)
    pd_int_pj = (pds + ints) / games if games > 0 else float("nan")

    return {
        "Pases defendidos":  pds,
        "Intercepciones":    ints,
        "Tackles":           tackles,
        "TFL":               tfl,
        "Fumbles forzados":  ff,
        "PD+INT / PJ":       pd_int_pj,
    }'''

BLOQUE_ONOFF_LBS = '''    gsis_id    = get_gsis_id(stats_row)
    team       = get_player_team(stats_row)
    epa_on_off = get_on_off_epa(gsis_id, team, pbp_df, part_df)

    return {
        "Tackles totales":  tackles,
        "TFL":              tfl,
        "Sacks":            sacks,
        "Intercepciones":   ints,
        "Pases defendidos": pds,
        "EPA on/off":       epa_on_off,
    }'''
NUEVO_LBS = '''    games = float(stats_row.get("games", 0) or 0)
    prod_pj = (tackles + tfl + sacks) / games if games > 0 else float("nan")

    return {
        "Tackles totales":  tackles,
        "TFL":              tfl,
        "Sacks":            sacks,
        "Intercepciones":   ints,
        "Pases defendidos": pds,
        "Producción/PJ":    prod_pj,
    }'''

BLOQUE_ONOFF_S = '''    gsis_id    = get_gsis_id(stats_row)
    team       = get_player_team(stats_row)
    epa_on_off = get_on_off_epa(gsis_id, team, pbp_df, part_df)

    return {
        "Tackles totales":       tackles,
        "Pases defendidos":      pds,
        "Intercepciones":        ints,
        "Blitz (QB hits+sacks)": blitz,
        "TFL":                   tfl,
        "EPA on/off":            epa_on_off,
    }'''
NUEVO_S = '''    games = float(stats_row.get("games", 0) or 0)
    impacto_pj = (tackles + pds) / games if games > 0 else float("nan")

    return {
        "Tackles totales":       tackles,
        "Pases defendidos":      pds,
        "Intercepciones":        ints,
        "Blitz (QB hits+sacks)": blitz,
        "TFL":                   tfl,
        "Impacto/PJ":            impacto_pj,
    }'''

BLOQUE_ONOFF_LINEA = '''    gsis_id = get_gsis_id(stats_row)
    team    = get_player_team(stats_row)
    epa_on_off = get_on_off_epa(gsis_id, team, pbp_df, part_df)

    return {
        "Sacks":            sacks,
        "QB hits":          qb_hits,
        "TFL":              tfl,
        "Fumbles forzados": ff,
        "EPA en sacks":     epa_sacks,
        "EPA on/off":       epa_on_off,
    }'''
BLOQUE_ONOFF_LINEA_DT = BLOQUE_ONOFF_LINEA.replace(
    "    gsis_id = get_gsis_id(stats_row)\n    team    = get_player_team(stats_row)\n",
    "    gsis_id    = get_gsis_id(stats_row)\n    team       = get_player_team(stats_row)\n")
NUEVO_LINEA = '''    games = float(stats_row.get("games", 0) or 0)
    disrupcion_pj = (sacks + qb_hits + tfl) / games if games > 0 else float("nan")

    return {
        "Sacks":            sacks,
        "QB hits":          qb_hits,
        "TFL":              tfl,
        "Fumbles forzados": ff,
        "EPA en sacks":     epa_sacks,
        "Disrupción/PJ":    disrupcion_pj,
    }'''


def cirugia_secundaria(f, bloque_viejo, bloque_nuevo, clave_vieja, clave_nueva,
                       fin_helpers, fin_data, firma_vieja, firma_nueva,
                       llamada_vieja, llamada_nueva):
    """cbs / lbs / safeties: fuera PBP + participation por completo."""
    src = leer(f)
    src = reemplazar(src,
        "from pbp_loader import cargar_pbp, cargar_stats, cargar_participation",
        "from pbp_loader import cargar_stats", "import")
    src = re.sub(r"MIN_SNAPS\s*=\s*100.*\n", "", src)
    src = borrar_seccion(src, r"^def get_gsis_id", fin_helpers, f"{f} helpers gsis")
    src = borrar_seccion(src, r"^# ── PARTICIPACIÓN ─", r"^def compute_", f"{f} seccion part")
    src = reemplazar(src, bloque_viejo, bloque_nuevo, f"{f} bloque onoff")
    src = reemplazar(src, firma_vieja, firma_nueva, f"{f} firma compute")
    src = borrar_seccion(src, r"^# ── DATA — PBP ─", r"^# ── FIND PLAYERS", f"{f} data pbp+part")
    src = re.sub(r'print\(f"  \[info\] GSIS IDs:.*\n', "", src)
    src = re.sub(r'print\(f"  \[info\] Teams:.*\n', "", src)
    src = reemplazar(src, llamada_vieja, llamada_nueva, f"{f} llamada compute")
    src = re.sub(r"nd_count = sum\(.*\n" + r'print\(f"  \[info\] EPA on/off.*\n', "", src)
    src = reemplazar(src, '    "EPA on/off",', f'    "{clave_nueva}",', f"{f} metric key")
    src = src.replace("Fuente: nflverse-data + stats_player + pbp_participation",
                      "Fuente: nflverse-data · stats_player")
    escribir(f, src)
    print(f, "OK")


def cirugia_linea(f, pos, bloque_viejo, fin_helpers, fin_find,
                  firma_vieja, firma_nueva, llamada_vieja, llamada_nueva):
    """edges / dts: mantienen PBP (EPA en sacks); fuera participation."""
    src = leer(f)
    src = reemplazar(src,
        "from pbp_loader import cargar_pbp, cargar_stats, cargar_participation",
        "from pbp_loader import cargar_pbp, cargar_stats", "import")
    src = re.sub(r"MIN_SNAPS\s*=\s*100.*\n", "", src)
    src = borrar_seccion(src, r"^def get_gsis_id", fin_helpers, f"{f} helpers gsis")
    src = borrar_seccion(src, r"^# ── PARTICIPACIÓN", r"^def compute_", f"{f} seccion part")
    src = reemplazar(src, bloque_viejo, NUEVO_LINEA, f"{f} bloque onoff")
    src = reemplazar(src, firma_vieja, firma_nueva, f"{f} firma compute")
    src = borrar_seccion(src, r"^# ── DATA — PARTICIPACIÓN", fin_find, f"{f} data part")
    src = re.sub(r'print\(f"  \[info\] GSIS IDs:.*\n', "", src)
    src = re.sub(r'print\(f"  \[info\] Teams:.*\n', "", src)
    src = reemplazar(src, llamada_vieja, llamada_nueva, f"{f} llamada compute")
    src = re.sub(r"(# Diagnóstico EPA on/off.*\n)?nd_count = sum\(.*\n"
                 + r'print\(f"  \[info\] EPA on/off.*\n', "", src)
    src = reemplazar(src, '    "EPA on/off",', '    "Disrupción/PJ",', f"{f} metric key")
    src = src.replace("Fuente: nflverse-data + stats_player + pbp_participation",
                      "Fuente: nflverse-data · stats_player + PBP")
    escribir(f, src)
    print(f, "OK")


cirugia_secundaria(
    "comparador_cbs.py", BLOQUE_ONOFF_CBS, NUEVO_CBS, "EPA on/off", "PD+INT / PJ",
    r"^def find_player", r"^# ── FIND PLAYERS",
    "def compute_cb_metrics(player_name, stats_row, pbp_df, part_df):",
    "def compute_cb_metrics(player_name, stats_row):",
    "    all_raw[name] = compute_cb_metrics(name, row.iloc[0], df_pbp, part_df)",
    "    all_raw[name] = compute_cb_metrics(name, row.iloc[0])")

cirugia_secundaria(
    "comparador_lbs.py", BLOQUE_ONOFF_LBS, NUEVO_LBS, "EPA on/off", "Producción/PJ",
    r"^def find_player", r"^# ── FIND PLAYERS",
    "def compute_lb_metrics(player_name, stats_row, pbp_df, part_df):",
    "def compute_lb_metrics(player_name, stats_row):",
    "    all_raw[name] = compute_lb_metrics(name, row.iloc[0], df_pbp, part_df)",
    "    all_raw[name] = compute_lb_metrics(name, row.iloc[0])")

cirugia_secundaria(
    "comparador_safeties.py", BLOQUE_ONOFF_S, NUEVO_S, "EPA on/off", "Impacto/PJ",
    r"^def find_player", r"^# ── FIND PLAYERS",
    "def compute_s_metrics(player_name, stats_row, pbp_df, part_df):",
    "def compute_s_metrics(player_name, stats_row):",
    "    all_raw[name] = compute_s_metrics(name, row.iloc[0], df_pbp, part_df)",
    "    all_raw[name] = compute_s_metrics(name, row.iloc[0])")

cirugia_linea(
    "comparador_edges.py", "edge", BLOQUE_ONOFF_LINEA,
    r"^def find_edge", r"^# ── FIND EDGE RUSHERS",
    "def compute_edge_metrics(player_name, stats_row, sack_df, sack_col, pbp_df, part_df):",
    "def compute_edge_metrics(player_name, stats_row, sack_df, sack_col):",
    "        name, row.iloc[0], sack_df, sack_col, df_pbp, part_df",
    "        name, row.iloc[0], sack_df, sack_col")

cirugia_linea(
    "comparador_dts.py", "dt", BLOQUE_ONOFF_LINEA_DT,
    r"^def find_player", r"^# ── FIND PLAYERS",
    "def compute_dt_metrics(player_name, stats_row, sack_df, sack_col, pbp_df, part_df):",
    "def compute_dt_metrics(player_name, stats_row, sack_df, sack_col):",
    "    all_raw[name] = compute_dt_metrics(name, row.iloc[0], sack_df, sack_col, df_pbp, part_df)",
    "    all_raw[name] = compute_dt_metrics(name, row.iloc[0], sack_df, sack_col)")

print("Cirugía completada.")
