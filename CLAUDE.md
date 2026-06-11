# NFL 2025 � @CuartayDato

An�lisis estad�stico NFL para redes sociales. Scripts Python independientes, generan PNGs.
Repo: https://github.com/luismi1702/NFL-2025

## Stack
pandas � numpy � xgboost � scikit-learn � matplotlib �nicamente (no Plotly, no Seaborn)
Datos nflverse: PBP, stats, games � cache en pbp_cache/ (parquet)
Logos en logos/{SIGLA}.png (ej: logos/SF.png)
Datos de participaci�n (FTN) disponibles 2016-2025: rutas, presi�n, coberturas, personal

## Carga de datos � usar SIEMPRE pbp_loader.py
from pbp_loader import cargar_pbp, cargar_stats, cargar_participation
df, SEASON = cargar_pbp(SEASON)   # SEASON=None auto-detecta; cache local; solo REG
- solo_reg=False solo en scripts de partido/semana concreta (resumen, previas, semanales)
- No usar pd.read_csv contra URLs de nflverse en scripts nuevos

## Estilo visual � obligatorio en todos los scripts
BG="#0f1115" � CARD="#151924" � FG="#EDEDED" � GRID="#2a2f3a" � ACCENT="#2d6cdf"
RYG=["#d84a4a","#ffd166","#06d6a0"]

Marca de agua � SIEMPRE esquina inferior derecha:
ax.text(0.99, 0.01, "@CuartayDato", transform=ax.transAxes, ha="right", va="bottom", color="#888888", fontsize=9, alpha=0.8, fontstyle="italic")
plt.savefig("output.png", dpi=200, bbox_inches="tight", facecolor=BG)

## Reglas
- OL es ATAQUE � nunca en grupos de defensa o trincheras
- CB y S van separados � no agrupar como "DB" (POS_MAP: "CB":"CB", "S":"S")
- set_yticklabels no acepta lista de colores � iterar sobre ax.get_yticklabels()
- Scripts independientes � no crear m�dulos compartidos salvo que se pida (�nica excepci�n: pbp_loader.py)
- Experimentos temporales van en lab/ (crearla si hace falta); solo lo definitivo vive en la ra�z
- Logos NYJ muy apaisados: reducir zoom �4.5 o �6.5
- SIEMPRE buscar web antes de escribir cualquier post � nunca asumir datos del modelo actualizados

## Posts X (@CuartayDato) � 280 chars m�ximo
Estructura: [historia del equipo] ? [dato como revelaci�n] ? [pregunta ilusionante ??] ? [#NFLDraft #Equipo]
- No empezar con n�meros en bruto
- No centrar en el QB salvo que el visual sea de QBs
- Cierre siempre ilusionante � el draft vende ilusi�n
- No usar "jugadores de franquicia"

## Docs (leer cuando se necesiten)
- Cat�logo de scripts: docs/scripts-catalog.md
- Cambios equipos NFL 2026: docs/nfl2026-changes.md
- Ejemplos de posts: docs/post-ejemplos.md
