# draft_success_r1.py
# Tasa de éxito en PRIMERA RONDA por posición y tramo de elección (2011-2022)
# Éxito = segundo contrato (≥2 años) con el mismo equipo que lo drafteó

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import nflreadpy

# === Config ===
SEASONS = list(range(2011, 2023))
BG      = "#0f1115"
CARD    = "#151924"
FG      = "#EDEDED"
ACCENT  = "#2d6cdf"
DPI     = 170
FIGSIZE = (14, 10)

RYG = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

# Tramos de elección (columnas del heatmap)
PICK_BINS   = [0, 5, 10, 15, 20, 25, 32]
PICK_LABELS = ["1–5", "6–10", "11–15", "16–20", "21–25", "26–32"]

POS_MAP = {
    "QB": "QB",
    "RB": "RB", "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "T":  "OL", "G": "OL", "C": "OL", "OL": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB",
    "CB": "DB", "S": "DB", "DB": "DB",
}
POS_ORDER = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"]

NICK_TO_ABBR = {
    "Cardinals":"ARI","Falcons":"ATL","Ravens":"BAL","Bills":"BUF","Panthers":"CAR",
    "Bears":"CHI","Bengals":"CIN","Browns":"CLE","Cowboys":"DAL","Broncos":"DEN",
    "Lions":"DET","Packers":"GB","Texans":"HOU","Colts":"IND","Jaguars":"JAX",
    "Chiefs":"KC","Rams":"LA","Chargers":"LAC","Raiders":"LV","Dolphins":"MIA",
    "Vikings":"MIN","Patriots":"NE","Saints":"NO","Giants":"NYG","Jets":"NYJ",
    "Eagles":"PHI","Steelers":"PIT","Seahawks":"SEA","49ers":"SF",
    "Buccaneers":"TB","Titans":"TEN","Commanders":"WAS",
    "Redskins":"WAS","Football Team":"WAS","Oilers":"TEN",
}
ABBR_NORM = {
    "OAK":"LV","LVR":"LV","SD":"LAC","STL":"LA","KAN":"KC","SFO":"SF","GNB":"GB",
    "TAM":"TB","JAC":"JAX","NOR":"NO","PHO":"ARI","HST":"HOU","NWE":"NE",
    "ARI":"ARI","ATL":"ATL","BAL":"BAL","BUF":"BUF","CAR":"CAR",
    "CHI":"CHI","CIN":"CIN","CLE":"CLE","DAL":"DAL","DEN":"DEN",
    "DET":"DET","GB":"GB","HOU":"HOU","IND":"IND","JAX":"JAX",
    "KC":"KC","LA":"LA","LAC":"LAC","LAR":"LA","LV":"LV","MIA":"MIA",
    "MIN":"MIN","NE":"NE","NO":"NO","NYG":"NYG","NYJ":"NYJ",
    "PHI":"PHI","PIT":"PIT","SEA":"SEA","SF":"SF","TB":"TB",
    "TEN":"TEN","WAS":"WAS",
}

def normalize_team(raw: str) -> str:
    if not isinstance(raw, str):
        return None
    base = raw.split("/")[0].strip()
    if base in ABBR_NORM:
        return ABBR_NORM[base]
    if base in NICK_TO_ABBR:
        return NICK_TO_ABBR[base]
    return None


# ─────────────────────────────────────────────
# 1. Datos
# ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print("Cargando draft picks 2011-2022...")
    draft = nflreadpy.load_draft_picks(seasons=SEASONS).to_pandas()
    draft["pos_group"] = draft["position"].map(POS_MAP)
    draft = draft[
        draft["pos_group"].notna() &
        draft["gsis_id"].notna()
    ].copy()
    draft["season"] = draft["season"].astype(int)
    draft["round"]  = draft["round"].astype(int)

    # Solo primera ronda
    draft = draft[draft["round"] == 1].copy()

    # Detectar columna de elección dentro de la ronda
    pick_col = next((c for c in ["pick", "pick_no", "draft_pick"] if c in draft.columns), None)
    if pick_col is None:
        raise SystemExit("No se encontró columna de número de pick en los datos.")
    draft["pick_num"] = pd.to_numeric(draft[pick_col], errors="coerce")
    draft = draft[draft["pick_num"].notna()].copy()

    # Asignar tramo
    draft["pick_group"] = pd.cut(
        draft["pick_num"],
        bins=PICK_BINS,
        labels=PICK_LABELS,
        right=True
    )

    print(f"Picks de primera ronda cargados: {len(draft)}")
    print(draft.groupby("pick_group", observed=True).size().to_string())

    print("Cargando contratos OTC...")
    c = nflreadpy.load_contracts().to_pandas()
    c = c.dropna(subset=["gsis_id","draft_team","team","year_signed","years","draft_year"]).copy()
    c["year_signed"] = c["year_signed"].astype(int)
    c["years"]       = c["years"].astype(float)
    c["draft_year"]  = c["draft_year"].astype(int)
    c["team_norm"]   = c["team"].apply(normalize_team)
    c["dt_norm"]     = c["draft_team"].apply(normalize_team)

    second = c[
        c["team_norm"].notna() & c["dt_norm"].notna() &
        (c["team_norm"] == c["dt_norm"]) &
        (c["year_signed"] >= c["draft_year"] + 3) &
        (c["years"] >= 2)
    ]["gsis_id"].unique()

    draft["success"] = draft["gsis_id"].isin(second).astype(int)
    return draft


def compute_heatmap(df: pd.DataFrame):
    rate  = df.groupby(["pos_group", "pick_group"], observed=True)["success"].mean().mul(100).unstack()
    count = df.groupby(["pos_group", "pick_group"], observed=True)["success"].count().unstack()
    rate  = rate.reindex(POS_ORDER).reindex(columns=PICK_LABELS)
    count = count.reindex(POS_ORDER).reindex(columns=PICK_LABELS)
    return rate, count


# ─────────────────────────────────────────────
# 2. Gráfico
# ─────────────────────────────────────────────
def plot_heatmap(rate: pd.DataFrame, count: pd.DataFrame):
    n_pos = len(POS_ORDER)   # 8
    n_col = len(PICK_LABELS) # 6

    CW, CH = 0.90, 0.78
    GX, GY = 0.14, 0.22
    SX, SY = CW + GX, CH + GY

    LEFT_MARGIN  = 1.5
    TOP_MARGIN   = 2.6
    BOTTOM_EXTRA = 0.9
    RIGHT_MARGIN = 1.0
    total_w = LEFT_MARGIN + n_col * SX + RIGHT_MARGIN
    total_h = TOP_MARGIN  + n_pos * SY + BOTTOM_EXTRA

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    # ── Título ──────────────────────────────────────────────────
    ax.text(LEFT_MARGIN, total_h - 0.25,
            "Tasa de éxito en el Draft NFL — Primera Ronda",
            ha="left", va="top",
            fontsize=20, fontweight="bold", color=FG, zorder=5)
    ax.text(LEFT_MARGIN, total_h - 1.05,
            "% de picks que firmaron un segundo contrato (≥2 años) con el mismo equipo que los drafteó  ·  Drafts 2011–2022",
            ha="left", va="top",
            fontsize=10, color="#888888", fontstyle="italic", zorder=5)

    # ── Cabeceras de tramo ───────────────────────────────────────
    for c, lbl in enumerate(PICK_LABELS):
        cx = LEFT_MARGIN + c * SX + CW / 2
        cy = total_h - TOP_MARGIN * 0.72
        pill = plt.Rectangle(
            (LEFT_MARGIN + c * SX, cy - 0.28), CW, 0.52,
            linewidth=0, facecolor=CARD,
            transform=ax.transData, zorder=1
        )
        ax.add_patch(pill)
        ax.text(cx, cy, f"Pick\n{lbl}",
                ha="center", va="center",
                fontsize=10, fontweight="bold", color=ACCENT, zorder=2,
                linespacing=1.2)

    # ── Celdas ──────────────────────────────────────────────────
    for r, pos in enumerate(POS_ORDER):
        row_y = total_h - TOP_MARGIN - (r + 1) * SY + GY / 2

        ax.text(LEFT_MARGIN - 0.18, row_y + CH / 2,
                pos,
                ha="right", va="center",
                fontsize=13, fontweight="bold", color=FG)

        for c, lbl in enumerate(PICK_LABELS):
            val = rate.loc[pos, lbl]  if not pd.isna(rate.loc[pos, lbl])  else np.nan
            n   = count.loc[pos, lbl] if not pd.isna(count.loc[pos, lbl]) else 0

            cell_color = RYG(val / 100) if not np.isnan(val) else "#1e2330"
            cx = LEFT_MARGIN + c * SX
            cy = row_y

            cell = mpatches.FancyBboxPatch(
                (cx, cy), CW, CH,
                boxstyle="round,pad=0.01,rounding_size=0.07",
                linewidth=0, facecolor=cell_color,
                transform=ax.transData, zorder=2
            )
            ax.add_patch(cell)

            if not np.isnan(val) and val >= 99:
                border = mpatches.FancyBboxPatch(
                    (cx - 0.025, cy - 0.025), CW + 0.05, CH + 0.05,
                    boxstyle="round,pad=0.01,rounding_size=0.09",
                    linewidth=2, edgecolor="#ffd700",
                    facecolor="none",
                    transform=ax.transData, zorder=3
                )
                ax.add_patch(border)

            if not np.isnan(val):
                ink     = "#0f1115" if val > 50 else FG
                ink_sub = "#1a2a1a" if val > 50 else "#666666"
                ax.text(cx + CW / 2, cy + CH * 0.60,
                        f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=15, fontweight="bold", color=ink, zorder=4)
                ax.text(cx + CW / 2, cy + CH * 0.25,
                        f"n={int(n)}",
                        ha="center", va="center",
                        fontsize=8, color=ink_sub, zorder=4)

    # ── Leyenda de color ─────────────────────────────────────────
    leg_y  = BOTTOM_EXTRA * 0.55
    leg_x0 = LEFT_MARGIN
    leg_w  = n_col * SX * 0.55
    leg_h  = 0.22
    n_seg  = 200
    for i in range(n_seg):
        seg_x = leg_x0 + i * (leg_w / n_seg)
        ax.add_patch(plt.Rectangle(
            (seg_x, leg_y), leg_w / n_seg + 0.01, leg_h,
            linewidth=0, facecolor=RYG(i / n_seg), zorder=2
        ))
    for pct, label in [(0, "0%"), (25, "25%"), (50, "50%"), (75, "75%"), (100, "100%")]:
        ax.text(leg_x0 + (pct / 100) * leg_w, leg_y - 0.12,
                label, ha="center", va="top", fontsize=8, color="#888888")
    ax.text(leg_x0 - 0.1, leg_y + leg_h / 2,
            "% éxito", ha="right", va="center", fontsize=8, color="#888888")

    # ── Fuente ──────────────────────────────────────────────────
    ax.text(LEFT_MARGIN, BOTTOM_EXTRA * 0.18,
            "Fuente: Pro Football Reference via nflreadpy  ·  n = nº de picks analizados por celda",
            ha="left", va="center", fontsize=7.5,
            color="#555555", fontstyle="italic")

    # ── Firma ───────────────────────────────────────────────────
    ax.text(total_w - 0.05, BOTTOM_EXTRA * 0.18,
            "@CuartayDato",
            ha="right", va="center",
            fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

    out = "draft_success_r1_por_eleccion.png"
    plt.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {out}")


# ─────────────────────────────────────────────
# 3. Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()

    print(f"\nTotal picks de 1ª ronda analizados: {len(df)}")
    print("\nTasa de éxito global por posición (%):")
    print(df.groupby("pos_group")["success"].mean()
            .mul(100).round(1)
            .reindex(POS_ORDER)
            .to_string())

    rate, count = compute_heatmap(df)
    plot_heatmap(rate, count)

    # ── Debug: jugadores específicos por posición y tramo ──────────
    pos_query   = input("\n¿Posición a inspeccionar (ej: QB, WR, vacío para saltar)? ").strip().upper()
    if pos_query:
        tramo_query = input("¿Tramo de picks (ej: 26–32, 1–5)? ").strip()
        name_col = next((c for c in ["pfr_player_name", "player_display_name", "player_name", "full_name", "name"] if c in df.columns), None)
        team_col = next((c for c in ["team", "team_abbr", "club_code"] if c in df.columns), None)
        show_cols = ["season", "pick_num", "position", "success"]
        if name_col: show_cols.insert(0, name_col)
        if team_col: show_cols.append(team_col)
        tramo_norm = tramo_query.replace("-", "–")   # normalizar guión corto → en-dash
        subset = df[
            (df["pos_group"] == pos_query) &
            (df["pick_group"].astype(str) == tramo_norm)
        ][show_cols].sort_values("pick_num")
        print(f"\n{pos_query} · Pick {tramo_query}  ({len(subset)} jugadores):")
        print(subset.to_string(index=False))
        renovaron = subset[subset["success"] == 1]
        if len(renovaron) > 0:
            print(f"\n  >> Renovaron ({len(renovaron)}):")
            for _, row in renovaron.iterrows():
                name = row[name_col] if name_col else "?"
                print(f"     {name}  ({int(row['season'])}, pick {int(row['pick_num'])}, {row.get(team_col, '')})")
        else:
            print("  >> Ninguno renovó.")

    rate, count = compute_heatmap(df)
    plot_heatmap(rate, count)
