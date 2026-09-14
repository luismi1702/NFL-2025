# NFL 2025 — @CuartayDato

Análisis estadístico NFL para redes sociales. Scripts Python independientes, generan PNGs.
Repo: https://github.com/luismi1702/NFL-2025

## Stack
pandas · numpy · xgboost · scikit-learn · matplotlib únicamente (no Plotly, no Seaborn)
Datos nflverse: PBP, stats, games — cache en pbp_cache/ (parquet)
Logos en logos/{SIGLA}.png (ej: logos/SF.png)
Datos de participación (FTN) disponibles 2016-2025: rutas, presión, coberturas, personal

## Carga de datos — usar SIEMPRE pbp_loader.py
from pbp_loader import cargar_pbp, cargar_stats, cargar_participation, cargar_ftn
SEASON = season_cli()             # respeta --season; None = auto-detecta
df, SEASON = cargar_pbp(SEASON)   # cache local; solo REG; avisa si hay poca muestra
- cargar_ftn: FTN charting (is_play_action, blitzers...) — solo 2022+
- solo_reg=False solo en scripts de partido/semana concreta (resumen, previas, semanales)
- No usar pd.read_csv contra URLs de nflverse en scripts nuevos
- Los datos auxiliares (participacion, FTN, stats) tardan dias o semanas en
  publicarse al arrancar la temporada. Envolver su carga en
  `try/except DatosNoDisponibles` explicando que necesita el visual
- AVISO: `pbp_participation` (cobertura, personal, rutas) NO tiene cron en
  nflverse — es un rebuild manual. En 2025 se ejecuto una sola vez, en febrero.
  No dar por hecho que esta fresco: `python estado_datos.py` lo comprueba
- Fuentes verificadas que SI se actualizan en temporada (cada 6h o a diario):
  cargar_pfr (cobertura CB/S, presiones, placajes fallados, pocket time),
  cargar_ngs (separacion, YAC sobre esperado, RYOE), cargar_snaps,
  cargar_lesiones, cargar_qbr (Total QBR de ESPN), cargar_stats_equipo,
  cargar_contratos (OverTheCap, para offseason)

## Salidas — nunca guardar con un nombre fijo
from pbp_loader import salida, season_cli, week_cli
plt.savefig(salida(f"mi_grafico_{SEASON}.png", SEASON), ...)
- salida() archiva en salidas/{año}/w{semana}/ y estampa _wNN en el nombre
- La semana por defecto es hasta donde llegan los DATOS (ultima_semana()), no
  una pedida por teclado; los scripts de semana concreta la pasan explicita
- Estampar `sello(SEASON)` en el pie del PNG ("NFL 2026 · datos hasta sem. 7")
- Todos los scripts aceptan --season / --week / --raiz via cli()

## Logos — recorte por tinta obligatorio
Normalizar por el area de tinta REAL, no por el lienzo: hay logos con mucho
margen transparente (NYJ es un wordmark 3768x1186 dentro de 4096x4096) que sin
recortar salen aplastados. Implementacion de referencia: load_logo en
ranking_wrs.py. Nunca reducir el zoom a mano para compensar.

## Estilo visual — obligatorio en todos los scripts
BG="#0f1115" · CARD="#151924" · FG="#EDEDED" · GRID="#2a2f3a" · ACCENT="#2d6cdf"
RYG=["#d84a4a","#ffd166","#06d6a0"]

Marca de agua — SIEMPRE esquina inferior derecha:
ax.text(0.99, 0.01, "@CuartayDato", transform=ax.transAxes, ha="right", va="bottom", color="#888888", fontsize=9, alpha=0.8, fontstyle="italic")
plt.savefig("output.png", dpi=200, bbox_inches="tight", facecolor=BG)

## Reglas
- OL es ATAQUE — nunca en grupos de defensa o trincheras
- CB y S van separados — no agrupar como "DB" (POS_MAP: "CB":"CB", "S":"S")
- set_yticklabels no acepta lista de colores — iterar sobre ax.get_yticklabels()
- Scripts independientes — no crear módulos compartidos salvo que se pida (única excepción: pbp_loader.py)
- Experimentos temporales van en lab/ (crearla si hace falta); solo lo definitivo vive en la raíz
- VERIFICACIÓN OBLIGATORIA antes de proponer un post: (1) cada número, contra el script que genera el visual, no contra el doc ni contra el PNG a ojo; (2) cada atribución ataque/defensa, contra la clasificación real de la métrica (sacks permitidos = ATAQUE); (3) cada nombre, récord, traspaso o resultado, con búsqueda web. Un post no se propone hasta que las tres pasen
- Antes de proponer una pieza nueva: la pregunta no es "¿falta esto en el
  catálogo?" sino "¿lo tiene alguien más?". Si el lector lo encuentra a un clic
  en cualquier web generalista, no aporta (así se descartó la clasificación, el
  parte de lesiones y el QBR). Ver docs/backlog.md
- SIEMPRE buscar web antes de escribir cualquier post — nunca asumir datos del modelo actualizados

## Posts X (@CuartayDato) — 280 chars máximo
Estructura (TEMPORADA, lo que se publica hoy): [gancho: historia, nombres propios] → [dato como revelación] → [cierre corto con opinión] → [#NFL | #Equipo | @cuentas]
Estructura (DRAFT, solo en la ventana de abril): [historia del equipo] → [dato como revelación] → [pregunta ilusionante 🤔] → [#NFLDraft #Equipo]
- No empezar con números en bruto
- No centrar en el QB salvo que el visual sea de QBs
- Cierre ilusionante: en draft es pregunta (el draft vende ilusión); en temporada, frase corta con opinión
- No usar "jugadores de franquicia"
- Tono humano, no robótico: el cierre ilusionante NO tiene por qué ser una pregunta. Prohibidas las preguntas retóricas de relleno ("¿Cuánto dura la era?", "¿Quién cierra la grieta?") — si el cierre no aporta, mejor terminar en el dato o en una frase corta con opinión
- Terminología: "3er down"/"4º down", nunca "bajada". Equipos y ciudades en inglés (New England, no Nueva Inglaterra)
- Menciones: SOLO de docs/cuentas-fans.md, nunca inventadas ni deducidas. Manda el tema del post, no el script del PNG. Máximo 2-3
- Si el ángulo del post es NEGATIVO para el equipo, no se etiqueta a su afición: se etiqueta para que compartan, y nadie comparte lo que certifica su caída
- Terminología de jugadas: "proteger al QB" / "mejorar la protección de pase" / "pass pro", NUNCA "proteger al pasador". "Forzar turnovers", NUNCA "robar balones"

## Docs (leer cuando se necesiten)
- Catálogo de scripts: docs/scripts-catalog.md
- Calendario de posts y automatizacion (batch, borradores, cola de copiar y pegar): docs/calendario-posts.md
- Cambios equipos NFL 2026: docs/nfl2026-changes.md
- Ejemplos de posts: docs/post-ejemplos.md
- Pendientes para cuando se suscriba a PFF+: docs/pff-wishlist.md
- Hilos de la fórmula del campeón (dos ángulos, ambos verificados): docs/hilo_formula_campeon.md (retrospectivo) y docs/hilo_deberes_2026.md (prescriptivo, 31 tuits, pendiente de publicar)
