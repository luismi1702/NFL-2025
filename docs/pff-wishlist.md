# PFF Wishlist — qué hacer cuando llegue la suscripción

DECIDIDO (Luis, 03-oct-2026): NO se paga ninguna suscripción por ahora. Se
empieza con fuentes GRATUITAS, y se pagaría PFF+ ($99,99/año; Pro de $199,99
descartado por precio) solo si una pieza concreta necesita algo que es solo
de PFF. SumerPass ($100) y FTN Stats iQ ($109,99) tampoco: lo que aportan en
temporada (coberturas y personal) está gratis en Sharp y SumerSports.

## Mapa: qué de esta lista se consigue gratis (comprobado el 03-oct-2026)

| Dato | Fuente gratuita | Nota |
|---|---|---|
| Win rate de cada liniero en protección y carrera (PBWR/RBWR) | ESPN, artículo de win rates 2026 | Sustituye al item 1 con otra medida: duelos ganados antes de 2,5 s, no presiones contadas |
| Pass rush win rate y % de dobles equipos por jugador | ESPN, mismo artículo | Item 7 |
| Presión cedida por OL (equipo, con/sin blitz, tiempo hasta lanzar) | Sharp, nfl-offensive-line-stats | Item 2 a nivel equipo |
| YPRR | StickToTheModel (estimado) y SumerSports (tablas de WR) | Item 4 |
| Cobertura CB/S (rating, recepciones permitidas) | PFR, `cargar_pfr` (ya en uso) | Item 6 |
| Placajes fallados | PFR (ya en uso) | Item 3 |
| Presiones por defensor | PFR (ya en uso) | |
| Coberturas por equipo (man/zone, middle open/closed) y USO de cada personal | Sharp | Personal y formación: uso Y resultado, ataque y defensa (SumerSports, cada noche, con EPA) y % de éxito por personal en ataque (Sharp, modo "Efficiency", los martes). Revisado a fondo el 05-oct-2026. Coberturas: solo uso y tasas. 02-nov-2026: tipo de cobertura (Cover 0-9) por equipo en 2026 y personal defensivo (Base/Nickel/Dime) con EPA SOLO de pago: StatRankings+ (34,99 $/mes, 139,99 $/año; Cover 0-9 por equipo y por jugador) y SūmerPass. statz.ai lo tiene con EPA pero de 2025; NFL Pro, bloqueado en España; SIS DataHub (gratis) va por jugador sin coberturas; Fantasy Points sin comprobar (pide sesión) |
| Grades PFF, BTT/TWP, presiones CONTADAS por liniero, bloqueo de carrera por hueco, lado/gap del rusher | Solo PFF | Items 5, 8, 9 y parte del 1 |

Las tablas web se leen y se copian a mano citando la fuente (sin scraping).
URLs en CLAUDE.md, "Carga de datos"

## Pendientes

1. **Presiones cedidas por liniero individual (LT/LG/C/RG/RT)** — jul-2026.
   El motivo original de la lista: saber si al LT le ganan más que al RT.
   No existe en datos públicos (intentado con nflverse: solo se puede atribuir
   el ORIGEN del rusher, no el bloqueador batido). Con el CSV de pass blocking
   de PFF, sustituir las flechas-proxy de `oline_presion_origen.py` por los
   datos reales por puesto — el diagrama de campo ya está construido.

2. **Pressure rate real como criterio de la fórmula del campeón** — may-2026.
   Descartada del análisis original por ser propietaria (ver docs/decisiones.md).
   `discriminacion_total.py` (el script que hizo ese análisis) se eliminó
   jul-2026 — la fórmula definitiva ya vive en `contenders_tracker.py`. Con
   suscripción: replicar la validación contra los 11 campeones 2015-2025 ahí
   y decidir si entra como criterio 13.

3. **Missed tackles** — may-2026. Ídem: candidata defensiva descartada por ser
   PFF. Validar como discriminador y, aparte, posible visual de tackling por
   equipo (estilo heatmap).

## Aprobados (jul-2026) — orden de ataque al suscribirse

4. **YPRR y targets por ruta (WR/TE)** — sustituir/complementar target share
   en ranking_wrs, ranking_tes, target_share y comparadores. La métrica de
   receptores del análisis moderno.
5. **Big-time throws vs turnover-worthy plays (QBs)** — pareja narrativa para
   posts y métrica extra del comparador_qbs.
6. **Passer rating permitido y recepciones cedidas por CB/S** — convertir
   comparador_cbs y comparador_safeties en comparadores de cobertura reales
   (hoy usan stats de conteo).
7. **Pass rush win rate y tasa de dobles equipos (edges/DTs)** — 7ª métrica de
   comparador_edges y comparador_dts; el DT doblado al 70% es un post entero.
8. **Run blocking grades por hueco** — cruzar con run_gap: EPA por hueco + 
   grade del bloqueo en ese hueco. Visual único.

## Script pendiente relacionado — dline_presion_origen.py (jul-2026)

Espejo defensivo de `oline_presion_origen.py`: qué presiones GENERA cada
equipo y quién las produce — por origen (interior/exterior/blitz LB/blitz DB)
y ranking de jugadores (presiones por rusher, por snap de pass rush).

Técnicamente la v1 no requeriría PFF (atribución de sacks + QB hits del PBP,
misma técnica que el de la OL cambiando posteam por defteam), pero DECISIÓN
del usuario (jul-2026): no hacerlo a medias — se construye en agosto directamente
con los datos completos de PFF (presiones con hurries incluidos y pass rush
win rate por jugador, item 7).

9. **Alineación del rusher: lado y gap** — ago-2026. Poder decir "de ese 9%
   de presión con blitz de secundario, un 5% entra por fuera del RT y un 4%
   por el hueco entre C y LG". Comprobado que NO existe en datos públicos:
   - `run_location`/`run_gap` del PBP solo existen para CARRERAS (10.298
     filas, todas play_type=run). `pass_location` es dónde se lanzó el balón,
     no por dónde llegó la presión.
   - `depth_chart_position` no distingue lado: da DE, OLB, DT, CB — nunca
     LDE/RDE ni LOLB/ROLB.
   - `participation.defense_positions` es la lista de quién está en el campo
     ("CB;CB;FB;ILB;ILB;K;OLB;OLB;RB;SS;TE"), sin alineación.
   - La descripción del sack nombra al autor y nada más.
   Es charting de alineación (técnica 0-9, lado). PFF lo tiene por snap de
   pass rush. Con ello, `dline_presion_origen.py` pasaría de 4 orígenes a
   orígenes × lado, y su espejo `oline_presion_origen.py` podría por fin
   decir el lado (hoy su pie dice "sin lado en los datos públicos").
   Es la extensión defensiva natural del item 1.

## Decisión de timing

Suscribirse a finales de AGOSTO 2026 (anual): un pago cubre temporada completa
+ playoffs + primavera de draft. Verificar antes si el dato college de
prospectos requiere la suscripción college aparte.

## Reglas

- CSVs siempre descargados a mano a `pff_data/` (licencia personal, no
  commitearlos: añadir `pff_data/` a .gitignore cuando exista).
- En los posts, citar "PFF" como fuente cuando el dato venga de ahí.

## Revisión con la prueba de PFF (09-nov-2026, PFF+ 7 días, cancelada el mismo día)

Revisado entero: informes por posición (34), por equipo, por partido y ficha de
jugador. Todo descarga CSV.

- NO tiene: tipo de cobertura (Cover 0-9) por equipo, personal defensivo
  (base/nickel/dime) ni EPA. Su "Coverage Scheme" es solo hombre/zona por
  defensor. Los huecos de SūmerPass del informe de equipo siguen sin cubrir.
- SÍ tiene, con columnas comprobadas:
  · Pase: BTT, BTT%, TWP, TWP%, ADJ%, drops, tiempo hasta lanzar, presión (item 5).
  · Recepción: Y/RR, rutas, slot/wide/inline, drops, contested catches (item 4).
  · Pass blocking por jugador: presiones cedidas (SK, HIT, HUR, PR), eficiencia,
    también en "true pass sets". La ficha "Snaps By Position" con "Detailed
    Positions" da snaps por puesto con LADO (LT/RT, TE-L/TE-R) → item 1 real.
  · Pass rush: presiones, PRP y WIN% (pass rush win rate), también en true pass sets (item 7).
  · Defensa: alineación por snaps (DL, BOX, SLOT, CNR, FS y A-gap/B-gap/over tackle/outside) → parte del item 9.
  · Run blocking: nota en esquema de zona y de gap (NO por hueco: item 8 solo a medias).
  · Por partido: emparejamientos defensor contra receptor (targets, yardas, INT).

## SumerLive / SumerBrain (prueba gratis, 09-nov-2026)

- La tabla de equipos (fan.sumersports.com/teams) trae Man %, Zone %, 2-High, 1-High y Base/Nickel/Dime % por defensa, sin Cover 0-9 ni botón de exportar. El filtro "Refinement" da el EPA con Base, Nickel o Dime (un filtro cada vez).
- Cover 0-6 por equipo CON EPA solo aparece preguntando a SumerBrain (su chat sobre la base de datos). De ahí salen `datos_web/sumerpass_coberturas_2026_w09.csv` (defensa y ataque) y `sumerpass_personal_defensivo_2026_w09.csv`, ya con el esquema del informe.
- Comprobado: el % de Base/Nickel/Dime de SumerBrain coincide al decimal con la tabla de equipos (CIN, LA, CLE, SEA).
- PFF: su API rechaza el acceso automático ("requires a PFF Pro subscription and an API key"). Solo vale el botón CSV pulsado a mano; Chrome dejó bajar uno (`pff_data/pff_offense_blocking_2026_w09.csv`).

## NFL Next Gen Trends (gratis, 12-nov-2026)

- nfl.com/stats/ngs/trends, lanzada por Next Gen Stats el 12-nov-2026 (tuit de Mike Band). Gratis y sin sesión.
- Por equipo: uso, puesto, percentil, EPA/jugada, % de éxito y % de big plays de 16 tendencias de ataque (personal 11/12/13/21, motion, play action en early downs, pase pedido, tempo, quick passing, rutas dentro/fuera, 10+ yardas de aire, bloqueo man/gap/inside zone/outside zone) y 12 de defensa (base/nickel/dime, man/zone/press, single-high/split-safety, Cover 0, blitz, presión con y sin blitz). En motion, play action, pase pedido y quick passing, también el rendimiento SIN la tendencia.
- Histórico de la liga desde 2018 por tendencia; por equipo, la evolución de sus 5 rasgos de identidad, temporada a temporada.
- Cubre gratis lo que el 02-nov se dio como solo de pago: personal defensivo con EPA y Cover 0. Sigue sin estar el desglose Cover 1-6 (eso, SumerBrain o StatRankings+).
- Copiador: `copiar_ngs_trends.py` -> `datos_web/ngs_trends_*` (tendencias, partidos, temporada y playoffs, entrenadores, medias de la liga desde 2018 e insights acumulados). Lo que va por API con token (temporadas anteriores por equipo, evolución año a año, dispersión) no se copia. El EPA es el del modelo de NGS (BUF w09: +0,107 frente a +0,140 en nflverse): no se mezcla con el nuestro.
- De pago (NFL Pro): los "Latest Insights" de jugadores.
