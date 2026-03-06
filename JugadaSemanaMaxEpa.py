# jugada_semana_max_wpa.py
# Texto + resumen: "La jugada con mayor WPA de la semana"
# Fuente: nflverse play_by_play_{SEASON} (descarga directa)
# Salida por consola en castellano (tweet sugerido + resumen + top3 opcional)

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

def safe_col(df, *names, default=None):
    """Devuelve la primera columna existente de 'names'."""
    for n in names:
        if n in df.columns:
            return n
    return default

def main():
    semana_str = input("Semana (número, p.ej. 5): ").strip()
    try:
        semana = int(semana_str)
    except:
        raise SystemExit("Semana inválida.")

    print(f"Descargando play-by-play {SEASON}...")
    df = pd.read_csv(URL, low_memory=False, compression="infer")

    if "week" not in df.columns:
        raise SystemExit("El dataset no tiene columna 'week'.")

    # Filtrar semana
    dfw = df[df["week"] == semana].copy()
    if dfw.empty:
        raise SystemExit(f"No hay jugadas para la semana {semana}.")

    # Columnas útiles con fallback
    desc_col   = safe_col(dfw, "desc", "play_description")
    playtype_c = safe_col(dfw, "play_type")
    passer_c   = safe_col(dfw, "passer")
    rusher_c   = safe_col(dfw, "rusher")
    receiver_c = safe_col(dfw, "receiver")
    qtr_c      = safe_col(dfw, "qtr")
    yardsg_c   = safe_col(dfw, "yards_gained")
    wpa_c      = safe_col(dfw, "wpa")
    epa_c      = safe_col(dfw, "epa")

    if not wpa_c:
        raise SystemExit("No se encontró columna de WPA en el dataset.")

    # Limpieza básica: excluir no_plays, kneels y spikes
    mask_real = dfw[wpa_c].notna()
    if playtype_c:
        mask_real &= ~dfw[playtype_c].astype(str).eq("no_play")
    for flag in ["qb_kneel", "qb_spike"]:
        if flag in dfw.columns:
            mask_real &= ~dfw[flag].fillna(False).astype(bool)

    plays = dfw[mask_real].copy()
    if plays.empty:
        raise SystemExit("No hay jugadas válidas tras filtrar penalidades/spikes/kneels.")

    # Ordenar por WPA (mayor primero)
    plays = plays.sort_values(wpa_c, ascending=False)

    # Top 1 (mayor WPA)
    top = plays.iloc[0]

    # Extraer info
    posteam = str(top.get("posteam", "NA"))
    defteam = str(top.get("defteam", "NA"))
    ptype   = str(top.get(playtype_c, "NA")) if playtype_c else "NA"
    wpa     = float(top[wpa_c])
    epa     = float(top.get(epa_c, np.nan)) if epa_c else np.nan
    yards   = float(top.get(yardsg_c, np.nan)) if yardsg_c else np.nan
    qtr     = int(top.get(qtr_c, np.nan)) if pd.notna(top.get(qtr_c, np.nan)) else None

    passer   = str(top.get(passer_c, "")) if passer_c else ""
    rusher   = str(top.get(rusher_c, "")) if rusher_c else ""
    receiver = str(top.get(receiver_c, "")) if receiver_c else ""
    desc     = str(top.get(desc_col, "")).strip() if desc_col else ""

    # Actor principal
    if ptype == "pass":
        actor = f"{passer} → {receiver}".strip(" → ")
    elif ptype == "run":
        actor = rusher if rusher else "carrera"
    else:
        actor = passer or rusher or receiver or ptype

    # Texto corto (tweet sugerido)
    qtr_txt = f"Q{qtr} · " if qtr else ""
    yards_txt = f"{int(yards)}y" if pd.notna(yards) else "—"
    epa_txt = f" · EPA {epa:+.3f}" if pd.notna(epa) else ""

    tweet = (
        f"[TWEET] Jugada con mayor impacto (WPA) - Semana {semana} NFL {SEASON}\n"
        f"{qtr_txt}{posteam} vs {defteam}: {actor} ({yards_txt}) "
        f"-> WPA {wpa:+.3f}{epa_txt}\n"
        f"@CuartayDato"
    )

    # Resumen extendido
    resumen = [
        "— Detalle —",
        f"Tipo: {ptype}",
        f"Equipo ataque: {posteam} · Rival: {defteam}",
        f"Yardas: {yards_txt}",
        f"WPA: {wpa:+.3f}{epa_txt}",
    ]
    if desc:
        resumen.append(f"Descripción: {desc}")

    # Imprimir
    print("\n" + "="*68)
    print(tweet)
    print("="*68)
    print("\n" + "\n".join(resumen))

    # Opcional: Top 3 jugadas WPA
    ver_top3 = input("\nMostrar TOP 3 jugadas por WPA de la semana? (s/n): ").strip().lower()
    if ver_top3 == "s":
        top3 = plays.head(3).copy()

        def fmt_row(i, row):
            t_ptype = str(row.get(playtype_c, "NA")) if playtype_c else "NA"
            t_post  = str(row.get("posteam", "NA"))
            t_def   = str(row.get("defteam", "NA"))
            t_wpa   = float(row[wpa_c])
            t_yards_raw = row.get(yardsg_c, np.nan) if yardsg_c else np.nan
            t_yards = f"{int(t_yards_raw)}y" if pd.notna(t_yards_raw) else "-"
            t_desc  = str(row.get(desc_col, "")).strip() if desc_col else ""
            if t_ptype == "pass":
                t_actor = f"{row.get(passer_c, '')} -> {row.get(receiver_c, '')}".strip(" -> ")
            elif t_ptype == "run":
                t_actor = str(row.get(rusher_c, "")) or "carrera"
            else:
                t_actor = str(row.get(passer_c, "")) or str(row.get(rusher_c, "")) or t_ptype
            suffix = f" | {t_desc}" if t_desc else ""
            return f"{i}) {t_post} vs {t_def}: {t_actor} ({t_yards}) -> WPA {t_wpa:+.3f}{suffix}"

        filas = [fmt_row(i, row) for i, (_, row) in enumerate(top3.iterrows(), 1)]
        print("\nTOP 3 WPA (semana):\n" + "\n".join(filas))

if __name__ == "__main__":
    main()
