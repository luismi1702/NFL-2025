# perfil_estructural_equipo.py
# Fuente: nflverse play_by_play_2025 (lectura online).

import io
import os
import ssl
import sys
import tempfile
import urllib.request

# Forzar UTF-8 en la terminal de Windows (evita UnicodeEncodeError con cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import certifi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Config ===
SEASON     = 2025
URL        = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"
CHUNK_SIZE = 1024 * 256  # 256 KB por chunk


# ---------------- FIX SSL + descarga por chunks (no carga todo en RAM) ----------------
def read_csv_gz_https(url: str) -> pd.DataFrame:
    """
    Descarga un .csv.gz por HTTPS en un archivo temporal (evita cargar todo en RAM)
    usando certificados de certifi. Evita SSL CERTIFICATE_VERIFY_FAILED en Windows.
    """
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with tempfile.NamedTemporaryFile(suffix=".csv.gz", delete=False) as tmp:
        tmp_path = tmp.name
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                tmp.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Descargando... {pct:.1f}%", end="", flush=True)
        print()

    try:
        df = pd.read_csv(tmp_path, compression="gzip", low_memory=False)
    finally:
        os.unlink(tmp_path)  # limpia siempre, incluso si pandas falla

    return df


# ---------------- Helpers ----------------
def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def success_rate_from_epa(epa: pd.Series) -> float:
    e = pd.to_numeric(epa, errors="coerce").dropna()
    return float((e > 0).mean() * 100) if len(e) else np.nan


def explosive_rate(sub: pd.DataFrame, exp_pass: int = 15, exp_run: int = 10) -> float:
    """% de jugadas explosivas (pase ≥15 yds ganadas o carrera ≥10 yds ganadas)."""
    y = pd.to_numeric(sub.get("yards_gained"), errors="coerce")
    pt = sub.get("play_type").astype(str)
    mask = ((pt == "pass") & (y >= exp_pass)) | ((pt == "run") & (y >= exp_run))
    valid = mask.dropna()
    return float(valid.mean() * 100) if len(valid) else np.nan


def get_week_max(df: pd.DataFrame) -> int:
    if "week" not in df.columns:
        return 0
    w = pd.to_numeric(df["week"], errors="coerce").dropna()
    return int(w.max()) if len(w) else 0


# ---------------- Normalización ----------------
def norm_0_1(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    mn, mx = x.min(), x.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        # Con un solo dato no hay rango: valor neutro 0.5
        return pd.Series([0.5] * len(x), index=x.index)
    scaled = (x - mn) / (mx - mn)
    return scaled if higher_is_better else (1 - scaled)


# ---------------- Métricas DEF ----------------
def defense_vs_coverage(df: pd.DataFrame, team: str, max_week: int | None):
    """
    Analiza la defensa por:
      1. Columna de cobertura real (si existe en el dataset)
      2. pass_location (izq/centro/der) — siempre presente en nflverse pbp
      3. down — último recurso
    Retorna (DataFrame, nombre_columna_usada).
    """
    sub = df[df["defteam"].astype(str) == team].copy()

    if max_week is not None and "week" in sub.columns:
        sub = sub[pd.to_numeric(sub["week"], errors="coerce") <= max_week]

    sub = sub[sub["play_type"].isin(["pass"])].copy()
    if sub.empty:
        return pd.DataFrame(), None

    sub = to_num(sub, ["epa"])

    # Prioridad de columna de agrupación
    cov_col = None
    for c in ["coverage", "coverage_scheme", "def_coverage", "coverage_type",
              "pass_location", "down"]:
        if c in sub.columns and sub[c].notna().any():
            cov_col = c
            break

    if cov_col is None:
        return pd.DataFrame(), None

    g = sub.groupby(cov_col, dropna=True, observed=True)
    out = pd.DataFrame(index=g.size().index)
    out["snaps"]       = g.size()
    out["EPA/jugada"]  = g["epa"].mean()
    out["Éxito (%)"]   = g["epa"].apply(success_rate_from_epa)

    return out.sort_values("snaps", ascending=False), cov_col


# ---------------- Métricas ATAQUE ----------------
def offense_vs_depth(df: pd.DataFrame, team: str, max_week: int | None):
    """
    Analiza el ataque por profundidad de pase (air_yards).
    Incluye EPA/jugada, tasa de éxito y % de jugadas explosivas.
    Agrupa directamente sobre datos crudos (evita mean-of-means).
    """
    if "air_yards" not in df.columns:
        return pd.DataFrame()

    sub = df[df["posteam"].astype(str) == team].copy()

    if max_week is not None and "week" in sub.columns:
        sub = sub[pd.to_numeric(sub["week"], errors="coerce") <= max_week]

    sub = sub[sub["play_type"].isin(["pass"])].copy()
    sub = to_num(sub, ["epa", "air_yards", "yards_gained"])

    sub["depth_bucket"] = pd.cut(
        sub["air_yards"],
        bins=[-1e9, 9.999, 19.999, 1e9],
        labels=["Cortos (<10)", "Medios (10–19)", "Profundos (20+)"]
    )

    # Agrupamos directamente sobre datos crudos → media ponderada correcta
    g = sub.groupby("depth_bucket", dropna=True, observed=True)
    out = g.agg(
        snaps=("epa", "size"),
        epa=("epa", "mean"),
        success=("epa", success_rate_from_epa),
    )

    # % explosivas por bucket calculado sobre datos crudos
    out["explosive_pct"] = [
        explosive_rate(sub[sub["depth_bucket"] == b])
        for b in out.index
    ]

    return out


# ---------------- Radar ----------------
def radar_plot(values: dict, labels: list, title: str, outfile: str):
    N = len(labels)
    if N < 3:
        print(f"  ⚠️  Solo {N} ejes disponibles; se necesitan ≥3 para el radar. Se omite.")
        return

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    vals = np.array([values.get(k, np.nan) for k in labels], dtype=float)

    # Detectar y avisar NaN antes de graficar
    nan_mask = np.isnan(vals)
    if nan_mask.any():
        bad = [labels[i] for i, m in enumerate(nan_mask) if m]
        print(f"  ⚠️  Ejes con NaN {bad}; se reemplazan por 0.")
        vals = np.where(nan_mask, 0.0, vals)

    # Cerrar polígono
    angles = np.concatenate([angles, [angles[0]]])
    vals   = np.concatenate([vals,   [vals[0]]])

    BG     = "#0f1115"
    CARD   = "#151924"
    INK    = "#EDEDED"
    SUB    = "#9aa6bd"
    GRID   = "#2b3140"
    ACCENT = "#4e9af1"

    fig = plt.figure(figsize=(7.5, 7.5), dpi=200)
    fig.patch.set_facecolor(BG)
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor(CARD)

    ax.plot(angles, vals, linewidth=2, color=ACCENT)
    ax.fill(angles, vals, alpha=0.25, color=ACCENT)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=INK, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["", "", "", ""], color=SUB)
    ax.grid(color=GRID, alpha=0.7)
    ax.set_title(title, color=INK, fontsize=14, fontweight="bold", pad=18)

    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✅ Guardado: {outfile}")


# ---------------- Main ----------------
def main():
    team = input("Equipo (siglas exactas, p.ej. SF): ").strip().upper()
    wk   = input("Hasta qué semana (Enter = toda la temporada): ").strip()

    max_week = None
    if wk != "":
        try:
            max_week = int(wk)
        except ValueError:
            raise SystemExit("Semana inválida. Ingresa un número entero.")

    print(f"\nCargando play-by-play {SEASON}...")
    df = read_csv_gz_https(URL)
    df = to_num(df, ["week", "epa", "yards_gained", "air_yards"])

    if max_week is None:
        max_week = get_week_max(df)

    # ──────────── DEFENSA ────────────
    print(f"\n── DEFENSA {team} (hasta semana {max_week}) ──")
    d_cov, cov_col_used = defense_vs_coverage(df, team, max_week)

    if d_cov.empty:
        print("⚠️  No se pudo calcular el perfil defensivo (faltan columnas necesarias).")
    else:
        print(f"   Agrupando por: '{cov_col_used}'")
        d_cov = d_cov[d_cov["snaps"] >= 50].copy()

        if d_cov.empty:
            print("⚠️  Ninguna categoría supera el mínimo de 50 snaps.")
        elif len(d_cov) < 2:
            print("⚠️  Menos de 2 categorías con datos suficientes; no se genera radar.")
        else:
            d_top = d_cov.head(6).copy()
            d_top["n_epa"]  = norm_0_1(d_top["EPA/jugada"], higher_is_better=False)
            d_top["n_succ"] = norm_0_1(d_top["Éxito (%)"],  higher_is_better=False)
            d_top["score"]  = d_top[["n_epa", "n_succ"]].mean(axis=1)

            labels_def = [str(x) for x in d_top.index]
            values_def = {lab: float(d_top.loc[lab, "score"]) for lab in d_top.index}

            radar_plot(
                values_def, labels_def,
                f"DEFENSA — {team} (hasta semana {max_week})\nAgrupado por {cov_col_used}",
                f"radar_DEFENSA_{team}.png"
            )
            print(d_top[["snaps", "EPA/jugada", "Éxito (%)"]].round(3))

    # ──────────── ATAQUE ────────────
    print(f"\n── ATAQUE {team} (hasta semana {max_week}) ──")
    o_depth = offense_vs_depth(df, team, max_week)

    if o_depth.empty:
        print("⚠️  No se pudo calcular profundidad de pase (falta air_yards).")
    else:
        o_depth = o_depth[o_depth["snaps"] >= 30].copy()

        if o_depth.empty:
            print("⚠️  Ningún bucket de profundidad supera el mínimo de 30 snaps.")
        elif len(o_depth) < 2:
            print("⚠️  Menos de 2 buckets con datos suficientes; no se genera radar.")
        else:
            o_depth["n_epa"]  = norm_0_1(o_depth["epa"],           higher_is_better=True)
            o_depth["n_succ"] = norm_0_1(o_depth["success"],        higher_is_better=True)
            o_depth["n_expl"] = norm_0_1(o_depth["explosive_pct"],  higher_is_better=True)
            o_depth["score"]  = o_depth[["n_epa", "n_succ", "n_expl"]].mean(axis=1)

            labels_off = [str(x) for x in o_depth.index]
            values_off = {lab: float(o_depth.loc[lab, "score"]) for lab in o_depth.index}

            radar_plot(
                values_off, labels_off,
                f"ATAQUE — {team} (hasta semana {max_week})\nProfundidad de pase",
                f"radar_ATAQUE_{team}.png"
            )
            print(o_depth[["snaps", "epa", "success", "explosive_pct"]].round(3))


if __name__ == "__main__":
    main()
