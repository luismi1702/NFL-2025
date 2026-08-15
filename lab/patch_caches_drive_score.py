"""
lab/patch_caches_drive_score.py
Parche puntual (jul-2026) tras la auditoria de scripts:

  1. Recalcula drive_score_rate con la definicion CORRECTA — agrupando drives
     por (week, fixed_drive) en vez de solo fixed_drive (que se reinicia cada
     partido y colapsaba ~180 drives de temporada en ~25) — y sobreescribe la
     columna en newmetrics_{yr}.parquet para 2015-2025.
  2. Recalcula third_conv_pct con la definicion oficial
     (third_down_converted / (converted + failed)) y la sobreescribe en
     situational_{yr}.parquet para 2015-2025.
  3. Imprime el drive_score_rate corregido de cada campeon 2015-2025 y el
     minimo, para recalibrar el umbral de contenders_tracker.py.

Usa los caches pbp_contenders_{yr}.parquet (sin descargas). Ejecutar desde la
raiz del proyecto:  python lab/patch_caches_drive_score.py
"""
import os
import numpy as np
import pandas as pd

CACHE   = "pbp_cache"
SEASONS = list(range(2015, 2026))


def metricas_corregidas(yr: int) -> pd.DataFrame:
    """drive_score_rate y third_conv corregidos por equipo para un año."""
    f = os.path.join(CACHE, f"pbp_contenders_{yr}.parquet")
    pbp = pd.read_parquet(f)
    pbp = pbp[pbp["season_type"] == "REG"]

    rows = []
    for team in pbp["posteam"].dropna().unique():
        off = pbp[pbp["posteam"] == team]
        # Un equipo juega como mucho 1 partido/semana → (week, fixed_drive)
        # identifica el drive sin necesidad de game_id.
        drives = off.groupby(["week", "fixed_drive"]).first()
        dsr = drives["drive_ended_with_score"].fillna(0).mean()

        tdc = off["third_down_converted"].sum()
        tdf = off["third_down_failed"].sum()
        tc  = tdc / (tdc + tdf) if (tdc + tdf) > 0 else np.nan

        rows.append({"team": team, "drive_score_rate_fix": dsr,
                     "third_conv_fix": tc, "n_drives": len(drives)})
    out = pd.DataFrame(rows)
    out["season"] = yr
    return out


def main():
    if not os.path.isdir(CACHE):
        raise SystemExit("Ejecutar desde la raiz del proyecto (falta pbp_cache/)")

    # Campeones desde schedules
    sch = pd.read_parquet(os.path.join(CACHE, "schedules.parquet"))
    sb  = sch[(sch["game_type"] == "SB") & sch["home_score"].notna()].copy()
    sb["champion"] = np.where(sb["home_score"] > sb["away_score"],
                              sb["home_team"], sb["away_team"])
    champions = sb[sb["season"].isin(SEASONS)].set_index("season")["champion"].to_dict()
    print("Campeones:", champions)

    champ_vals = {}
    for yr in SEASONS:
        fix = metricas_corregidas(yr)

        # ── newmetrics: drive_score_rate
        nm_path = os.path.join(CACHE, f"newmetrics_{yr}.parquet")
        nm = pd.read_parquet(nm_path)
        antes = nm["drive_score_rate"].mean()
        nm = nm.merge(fix[["team", "drive_score_rate_fix"]], on="team", how="left")
        nm["drive_score_rate"] = nm["drive_score_rate_fix"].fillna(nm["drive_score_rate"])
        nm = nm.drop(columns=["drive_score_rate_fix"])
        nm.to_parquet(nm_path, index=False)

        # ── situational: third_conv_pct
        sit_path = os.path.join(CACHE, f"situational_{yr}.parquet")
        sit = pd.read_parquet(sit_path)
        sit = sit.merge(fix[["team", "third_conv_fix"]], on="team", how="left")
        sit["third_conv_pct"] = sit["third_conv_fix"].fillna(sit["third_conv_pct"])
        sit = sit.drop(columns=["third_conv_fix"])
        sit.to_parquet(sit_path, index=False)

        champ = champions.get(yr)
        cv = fix.loc[fix["team"] == champ, "drive_score_rate_fix"]
        cv = float(cv.iloc[0]) if len(cv) else np.nan
        champ_vals[yr] = (champ, cv)
        media = fix["drive_score_rate_fix"].mean()
        nd = fix["n_drives"].mean()
        print(f"  {yr}: dsr medio liga {antes:.4f} -> {media:.4f} "
              f"(~{nd:.0f} drives/equipo) | campeon {champ}: {cv:.4f}")

    print("\nDrive score rate corregido de los campeones:")
    for yr, (champ, cv) in sorted(champ_vals.items()):
        print(f"  {yr} {champ}: {cv:.4f}")
    vals = [cv for _, cv in champ_vals.values() if not np.isnan(cv)]
    print(f"\nPeor campeon: {min(vals):.4f}  ->  umbral sugerido "
          f"(con margen): {min(vals) - 0.002:.4f}")


if __name__ == "__main__":
    main()
