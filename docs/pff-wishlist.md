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
| Coberturas por equipo (man/zone, middle open/closed) y USO de cada personal | Sharp | Personal: uso Y resultado (EPA, yardas y % de éxito por agrupación, modo "Efficiency" de Sharp; corregido el 05-oct-2026). Coberturas: solo uso y tasas. SumerSports en abierto no se actualiza |
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
