# CHANGELOG — NFL 2025 @CuartayDato

---

## [2026-09-14] — El cache servía jornadas a medias

**Qué se hizo:**
- **Bug encontrado el lunes de la semana 1**: el cache del PBP tenía 2 partidos
  (miercoles y jueves) y `estado_datos.py` decía "al dia". La liga llevaba 15.
  La causa: solo se re-descargaba si el calendario mostraba una semana
  POSTERIOR a la del cache, y el jueves y el domingo son la misma semana
- `_info_schedules()` devuelve ahora un tercer valor, los partidos REG jugados,
  y `cargar_pbp` se refresca si le faltan partidos aunque la semana coincida.
  Nuevo helper `partidos_jugados(season)`
- Mismo arreglo en `_cargar_auxiliar` (tenía el punto ciego equivalente: solo
  refrescaba con el cache a 3+ días). `stats_team` tenia 2 partidos con 15
  publicados. El chequeo por partidos exige 6 h de antigüedad para no
  re-descargar en bucle una fuente retrasada de origen
- `estado_datos.py` cuenta partidos además de semanas en las fuentes con
  `game_id`, y normaliza `season_type` ("REG" en el PBP, "Regular" en el QBR de
  ESPN — la comparación estricta daba un falso "faltan 15 partidos")
- Verificado: cache falseado a 2 partidos → se re-descarga solo y quedan 15;
  temporadas pasadas no disparan descargas; segunda llamada en 0,1 s
- Generados los resúmenes de CHI 59-37 CAR (récord: 96 puntos, los más de una
  semana 1) y DET 31-30 NO (prórroga), con 4 borradores de posts
- Generados los 15 partidos de la jornada: 30 PNGs en `salidas/2026/w01/`
- **Los resúmenes se nombran por orden de kickoff**: `NN_resumen_VIS_vs_LOC`,
  con NN del calendario (01 el inaugural del miércoles, 15 el Sunday Night).
  Helper nuevo `orden_partido(season, week, *equipos)` en `pbp_loader`, que
  ademas devuelve visitante/local, asi que el nombre ya no depende del orden en
  que se teclean las siglas (convivian `resumen_LA_vs_SF` y `resumen_SF_vs_LA`
  del mismo partido). Los 30 ficheros de la w01 renombrados
- **Los resumenes entran en el batch del martes**, y de TODOS los partidos:
  paso nuevo `resumenes()` en `semana_auto.py`, entre el dato de la semana y
  los power rankings. Generarlos no es editorial; elegir cual se publica si.
  Un partido caido no tumba a los demas. Probado sobre la w01: 15/15 en ~3 min

**Archivos modificados:** `pbp_loader.py`, `estado_datos.py`,
`resumen_partido.py`, `semana_auto.py`, `docs/scripts-catalog.md`,
`docs/calendario-posts.md`, `docs/decisiones.md`, `CHANGELOG.md`

**Pendiente:**
- PFR semanal, snap counts y QBR van genuinamente retrasados en nflverse (2 de
  15 partidos tras forzar la descarga): repetir el jueves
- Sigue sin decidir lo del QB destacado de la ficha (ahora los Saints muestran
  a T.Shough -1.0 pese a sus 410 yardas)

---

## [2026-09-11] — Semana 1 de 2026: resúmenes de los dos primeros partidos

**Qué se hizo:**
- Generados los 4 PNGs de `resumen_partido.py` de la semana 1: SEA 13-10 NE
  (Kickoff Game) y SF 27-7 LA (MCG de Melbourne, primer partido oficial en
  Australia)
- **`resumen_partido.py` acepta siglas alternativas**: `LAR`→`LA` (los Rams son
  "LA" en nflverse), más `JAC`, `WSH`, `LVR`, `GNB`, `KAN`… Antes `LAR` moría con
  "No se encontro el partido" desde la galería web. El error ahora lista los
  partidos disponibles de esa semana
- **Norma de las "Claves del partido" con respaldo**: con menos de 3 partidos en
  la temporada en curso usa la regular anterior entera, rotulado en subtítulo y
  pie. En semana 1 los dos PNGs salían vacíos ("Sin desviaciones con muestra
  suficiente")
- Borradores de 4 posts (uno positivo y otro negativo por partido), todos con
  los números impresos en los PNGs. Descartada una primera ronda que era resumen
  de lo ocurrido: sin dato propio no aporta nada frente a cualquier web
- **Presiones de los 49ers: no hay dato público todavía.** Del PBP salen 8 golpes
  al QB y 0 sacks en 28 dropbacks (4 de Odighizuwa); las presiones con *hurries*
  necesitan charting (FTN sin publicar, PFR publicado pero vacío). Ni PFF ni
  49ers Webzone ni SI dan el total

**Archivos modificados:** `resumen_partido.py`, `CHANGELOG.md`,
`docs/decisiones.md`, `docs/scripts-catalog.md`

**Pendiente:**
- La ficha del partido elige como QB destacado al de mejor EPA: en los Rams sale
  "S.Bennett +0.2", que solo jugó el último drive, con Stafford (15/25, 155 yds)
  en negativo. Propuesto cambiarlo al QB con más jugadas de pase; sin decidir
- Repetir los resúmenes cuando FTN y PFR publiquen 2026: añaden la faceta de
  presión, que ahora se cae de las claves
- Sin publicar ninguno de los 4 posts

---

## [2026-08-26] — Primer contacto con un narrador de la NFL en España

**Qué se hizo:**
- Redactado y **enviado** el mail de presentación a Pepe (narrador de NFL en
  español, aficionado de Buffalo). Objetivo: solo presentarse, sin pedir nada;
  la oferta latente es un documento de previa/post-partido por encuentro
- Como muestra van tres PNGs ya generados de `salidas/2025/w18/`:
  `informe_ataque_BUF`, `informe_defensa_BUF` y `ataque_por_personal_ofensivo`
- **Dato verificado** contra `ataque_por_personal_ofensivo_2025_w18.png`:
  Buffalo fue el mejor de los 32 equipos con personal 13 (+0.394 EPA), por
  delante de los Rams (+0.234). Corregido en el mail, que decía solo "más que
  los Rams". Salvedad conocida: BUF lo usó 50 snaps y LAR 331
- Otro hallazgo del informe de BUF, no usado en el mail pero sí anotado: nº1 de
  la NFL en EPA/carrera y a la vez peor hueco de la liga por el lado del tackle
  izquierdo (-0.40 EPA, #32)
- Correcciones de estilo al borrador del usuario: "personal 13" en vez de
  "posiciones de 3 TEs", comas, tildes y signos de apertura

**Archivos modificados:** ninguno (el mail se escribió y envió fuera del repo)

**Pendiente:**
- Esperar respuesta. Si contesta, preparar una previa real de un partido
- Los informes adjuntos son de la temporada 2025; sin comprobar si Buffalo movió
  la línea ofensiva este offseason, cosa que cambiaría la lectura del hueco del LT

---

## [2026-08-24] — Cola de copiar y pegar, estilo alineado y cierres del hilo

**Qué se hizo:**
- **`cola_posts.py` nuevo (nivel 2.5)**: convierte `borradores_posts.md` en
  `cola_posts.html` — tarjeta por alternativa, PNG visible, botón Copiar y
  caracteres recontados en Python (cruce contra lo que declara el redactor).
  Lo lanza el batch del martes tras los borradores. Descartada la API de X
- **Formato de borradores como contrato**: `borradores_prompt.md` fija sección,
  línea `IMAGEN:` y cada alternativa en un bloque cercado, para que el parser
  no dependa de markdown libre. Si el redactor se sale, la página lo dice
- Verificado con clics reales: copiar→pegar devuelve el texto exacto, y también
  por el camino de `execCommand` que es el que corre al abrir con doble clic
- **`docs/post-ejemplos.md` reescrito** como referencia de temporada, con tres
  ejemplos buenos y uno flojo comentado; los de draft quedan aparte y marcados.
  De paso, de cp1252 a UTF-8
- **`CLAUDE.md`**: estructura de temporada separada de la de draft, que se
  contradecían; línea de menciones; y restaurados 39 caracteres perdidos (U+FFFD)
- **Regla nueva**: si el ángulo del post es negativo para el equipo, no se
  etiqueta a su afición. Propagada a CLAUDE.md, el prompt, cuentas-fans y ejemplos
- **Hilo "Deberes 2026" cerrado**: los 31 tuits con `#NFL | #Equipo | @cuentas`,
  ninguno pasa de 280 (221-276), conteos recalculados. Sin tocar ningún dato
- Limpieza de la tabla de cuentas: fuera @9MICHEL9, @TomasTDN y @PabloFR_

**Archivos modificados:** cola_posts.py (nuevo), semana_auto.py, borradores_prompt.md, CLAUDE.md, docs/post-ejemplos.md, docs/calendario-posts.md, docs/cuentas-fans.md, docs/scripts-catalog.md, docs/hilo_deberes_2026.md

**Pendiente:**
- La primera corrida real del martes dirá si el redactor respeta el contrato de
  formato — hasta la semana 1 de 2026 no hay con qué probarlo
- ATL e IND siguen sin cuenta de aficionados: sus tuits del hilo van sin mención
- El batch del sábado seguirá fallando cada semana hasta que haya datos de 2026

---

## [2026-08-21] — Calendario de posts y nivel 1 de automatizacion

**Qué se hizo:**
- **Calendario semanal decidido** (docs/calendario-posts.md): martes dato +
  resumen, miercoles power rankings + MVPs, jueves bot con balance publico y
  gancho del TNF, domingo por la mañana el HILO de la jornada con una previa
  por partido, tematica quincenal. Viernes y sabado sin publicar
- **`semana_auto.py` nuevo**: batch de generacion (la maquina genera, Luis
  revisa y publica — la verificacion triple no se automatiza). Dos tareas de
  Windows registradas: martes 8:00 y sabado 23:00, con StartWhenAvailable.
  Log en salidas/auto_log.txt; un paso caido no arrastra al resto
- **Previas.py: deteccion de partidos desde el calendario** (cargar_calendario)
  en modo jornada — desde el PBP era una trampa: el sabado la semana N solo
  tiene el TNF y el hilo salia con un solo partido. Verificado: 16 partidos,
  16 PNGs + PDF
- Verificado el batch completo con la semana 18 de 2025; los pasos del bot
  avisan correctamente de que 2026 no ha empezado
- Decision de marca: pronosticos del bot PUBLICOS con balance honesto

**Archivos modificados:** semana_auto.py (nuevo), Previas.py, docs/calendario-posts.md (nuevo), .gitignore, docs/scripts-catalog.md

---

## [2026-08-21] — Tres mejoras de tarjetas: origen cruzado, WP con línea y pies honestos

**Qué se hizo:**
- **matchup_intel: sección nueva ORIGEN DE LA PRESIÓN** — el cruce del backlog:
  por dónde genera presión la defensa (PFR real) contra por dónde la cede el
  ataque (atribución sacks+hits), por 100 dropbacks con media de liga y rank
  por origen. Badge RIESGO/EXPLOIT con umbral de desvío conjunto ±2,5. Fuera
  del Top-3 (unidades distintas al EPA). Los números cuadran con los informes
  de equipo y con los dos scripts de origen
- **resumen_partido: la curva de WP usa `vegas_home_wp`** (y `vegas_wpa` para
  el Top-3), con fallback a la neutral. El favorito arranca en su probabilidad
  pregame, no en 50 %; el título dice qué versión dibuja
- **informe_equipo: pie del bloque PRESIÓN honesto por cara** — en defensa las
  flechas suman el KPI (misma fuente PFR, "n=158 presiones"); en ataque son
  atribución sacks+hits que NO suma el KPI de arriba y ahora lo dice
  ("n=98 sacks+hits"). Cierra el pendiente del n= ambiguo
- **informe_equipo: media de liga en las claves de hueco** (commit anterior
  d667087) — un +0.06 interior con liga en -0.04 ya se explica solo

**Archivos modificados:** matchup_intel.py, resumen_partido.py, informe_equipo.py, docs/backlog.md

---

## [2026-08-21] — Manning Bot v6: alcanza al mercado (68,9 % vs 68,2 %)

La bateria de experimentos de la mañana encontro dos palancas que suman, y
juntas cierran el hueco con la linea de apuestas que la v5 nunca cerro.

**Qué se hizo:**
- **Regresion del margen de puntos** (XGBRegressor+Ridge → probabilidad via
  normal) promediada con el clasificador de siempre. El 27-24 enseña mas que
  el gano/perdio: +1,7 pts ella sola
- **7 features de estabilidad**: success rate (of/def), EPA en situacion
  neutral (wp 5-95 %, fuera garbage time), EPA de pase en downs 1-2 (of/def)
  y EPA de equipos especiales — fase que el bot ignoraba. +1,7 pts solas
- **Resultado walk-forward 2022-2025**: 68,9 % vs 68,2 % del mercado (v5:
  65,7 %), ganandole en 3 de 4 temporadas; en discrepancias pasa de acertar
  el 37,5 % al 54,5 % (n=77, no venderlo como ventaja sistematica). Robusto a
  la semilla (68,2-69,0 %). El Brier global sigue una pizca peor (0,2112 vs
  0,2083); en 2024 tambien lo gana
- Descartado con numeros: calibracion isotonica, mezcla bot+mercado, poda de
  features y el parte de lesiones de QB (el bot ya sabe quien titula:
  schedules trae home_qb_id/away_qb_id). Detalle en docs/backlog.md
- **Bug pre-temporada arreglado**: la reconstruccion de game logs intentaba
  descargar el PBP de la temporada a predecir, que antes del kickoff no
  existe (404). Ahora se omite si no hay partidos jugados
- Caches pbp_{season}.parquet migrados desde los pbp_full locales (las 4
  columnas nuevas sin re-descargar nada)
- Modelo reentrenado y guardado (`ModeloManningV6`, 39 features). Ojo: los
  .pkl de v5 no cargan con el codigo v6 — reentrenar en vez de cargar

**Archivos modificados:** Manning_bot.py, docs/backlog.md, docs/scripts-catalog.md, manning_bot_model.pkl, lab/manning_exp_bateria.py (nuevo)

---

## [2026-08-21] — Manning Bot reentrenado y medido contra el mercado

**Qué se hizo:**
- **Modelo reentrenado con 2015-2025** (antes iba hasta 2024). Walk-forward
  medio 0,655 de acierto; `home_impl_prob` sigue siendo la feature dominante
  (0,118, casi el triple que la siguiente). `manning_bot_model.meta` ya no
  avisa de caducidad
- **`--bench` nuevo: el bot contra la línea de apuestas.** Walk-forward
  2022-2025, 953 partidos. **El bot pierde**: 65,7 % vs 68,2 % de aciertos y
  peor Brier en las cuatro temporadas. Discrepa del mercado en el 10,1 % de
  los partidos y ahí acierta solo el **37,5 %** — la discrepancia es ruido, no
  ventaja. No se publica como "bot vs Vegas" ganador; queda como línea base
  interna que cualquier mejora del modelo tiene que superar
- No hizo falta `vegas_wp` para esto: el moneyline de schedules cubre el 100 %
  de los partidos desde 2015 y es la misma información
- **Flags para correr sin teclado**: `--retrain` / `--no-retrain` / `--week N`
  / `--bench`. Antes el script paraba en dos `input()` y no se podía programar
- **Bug latente arreglado en `load_pbp`**: el caché `pbp_{season}.parquet` se
  daba por bueno solo por existir. En disco había ficheros de una versión
  anterior con 5 columnas en vez de 19, así que la primera reconstrucción de
  game logs habría reventado con un `KeyError`. Ahora se valida el esquema y
  se vuelve a descargar si faltan columnas

- **Experimentos de mejora, medidos contra la línea base** (scripts en `lab/`,
  probs walk-forward reutilizables en `lab/manning_probs_wf.parquet`):
  - Diagnóstico: el hueco con el mercado está TODO en semanas 5-17 (65,7 % vs
    68,8 %); en semanas 1-4 y 18 empata. Es información de alineación, no matemática
  - Calibración isotónica: empeora el Brier. Descartada
  - Mezcla bot+mercado: el mejor peso da +0,5 pts (ruido) con peor Brier. Descartada
  - `d_qb_out` (titular Out/Doubtful del parte de lesiones): señal real (el
    equipo sin titular gana el 31,3 %) pero cubre solo el 1,3 % de team-weeks
    (los IR no salen en el parte semanal) y el modelo empeora. No entra en producción
  - Pista abierta: en las 7 discrepancias fuertes (>60 % confianza) el bot
    acertó el 57 % — muestra mínima, vigilar en 2026

**Archivos modificados:** Manning_bot.py, docs/backlog.md, manning_bot_model.pkl, manning_bot_model.meta, lab/manning_experimentos.py (nuevo), lab/manning_exp_qb_out.py (nuevo)

**Pendiente:**
- Sigue vivo: comprobar `pbp_participation_2026` en la semana 2 con `python estado_datos.py`
- Sigue vivo: confirmar en vivo que `injuries` se actualiza
- Sigue vivo: lanzar `python generar_caches.py 2026` tras la semana 4
- Sigue vivo: el `n=` del bloque de presión del informe significa cosas distintas en cada cara con la misma etiqueta

---

## [2026-08-16] — Espejo defensivo de la presión y rediseño del informe de equipo

**Qué se hizo:**
- **`dline_presion_origen.py` nuevo**: desde dónde genera presión cada defensa, con heatmap de los 32 y diagrama de campo con flechas convergiendo sobre el QB. Se destrabó porque `pfr_advstats` da presiones reales (con hurries), que era lo que faltaba en julio
- **Clasificación edge/interior corregida**: ni PFR ni nflverse valen (Parsons es "DL" y "LB" respectivamente). Se usa `depth_chart_position` + regla de peso (DE ≥280 lb = interior). Sin ella, GB salía como defensa de interior cuando es lo contrario
- **Cambio de medida en los tres scripts de presión**: de reparto porcentual a **tasa por 100 dropbacks**. El reparto engañaba — SF salía "60% exterior vs 49% de liga" cuando su tasa exterior está por debajo de la media
- **`informe_equipo.py` rediseñado**: cabecera comprimida, 3 KPIs, DOMINA y SUFRE simétricos en banda, franja de carrera por hueco al pie y bloque PRESIÓN unificado con mini-campo
- **DOMINA/SUFRE con 19 candidatos ocultos** (zona roja, downs, explosivas, turnovers, play-action, clutch, tendencia últimas 4, success rate, sacks, penalizaciones, two-minute, blitz, presión y huecos): solo salen si el equipo es extremo, para que el resumen aporte algo que no está dibujado
- **Cuatro radares con la métrica más redundante sustituida**, elegida midiendo correlación: QBs (pocket limpio r=0.905 con EPA global → % EPA de aire), WRs (tasa recepción r=-0.747 con aDOT → separación NGS), edges y DTs (disrupción/PJ r=0.933 con sacks → presiones/PJ reales)
- **`clasificacion.py` construido y borrado el mismo día** por decisión editorial: el dato está en cualquier web
- Verificado: 32 equipos generan sin fallos y sin problemas de contenido

**Archivos modificados:** dline_presion_origen.py (nuevo), oline_presion_origen.py, informe_equipo.py, comparador_qbs/wrs/edges/dts/cbs/safeties.py, pbp_loader.py, estado_datos.py (nuevo), docs/backlog.md (nuevo), docs/pff-wishlist.md, docs/decisiones.md, docs/scripts-catalog.md

**Pendiente:**
- **Reentrenar Manning Bot** antes de la jornada 1 (el `.pkl` es de 2015-2024; el script ya avisa)
- **Comprobar `pbp_participation_2026` en la semana 2** con `python estado_datos.py` — no tiene cron y de él dependen 13 scripts
- Confirmar en vivo que `injuries` se actualiza (el workflow localizado no corre desde ago-2025)
- Lanzar `python generar_caches.py 2026` tras la semana 4
- El `n=` del bloque de presión del informe significa cosas distintas en cada cara con la misma etiqueta

---

## [2026-08-15] — Auditoría de datos: 7 fuentes nuevas y un aviso serio

La auditoría del día anterior fue centrada en el código y no revisó qué datos
hay disponibles. Luis lo señaló. Al hacerlo bien aparecieron dos cosas grandes.

**`pfr_advstats` cubre media wishlist de PFF, gratis.** 190 assets de Pro
Football Reference, actualizados cada 6 h en temporada, sin usar en el
proyecto. Cubre los items 3 y 6 de `docs/pff-wishlist.md` exactamente como
estaban especificados: placajes fallados y cobertura por defensor (`tgt`,
`cmp_percent`, `yds_tgt`, `rat`, `dadot`). También presiones por jugador
(`prss`/`hrry`/`qbkd`), que la wishlist daba por imposible sin pagar. Siguen
siendo de PFF los items 1 y 8 (presión por liniero, bloqueo por hueco).

**`pbp_participation` no tiene cron.** Es `workflow_dispatch` manual: 8
ejecuciones en el historial, UNA durante la temporada 2025 (10-feb-2026, ya
acabada, rellenando 22 semanas de golpe). De ahí salen cobertura, personal,
presión y rutas → 13 scripts. `ftn_charting` sí corre cada 6 h pero no tiene
esas columnas, así que no sustituye.

**Cadencia verificada** leyendo los `cron` de nflverse-pfr, ngs-data,
nflverse-rosters y nflverse-ftn, y contando ejecuciones de la temporada 2025:
pfr_advstats 38 · snap_counts 40 · nextgen_stats 72 · participation 1.

**Nuevo:**
- 7 cargadores en `pbp_loader`: `cargar_pfr` (def/pass/rec/rush, acumulado o
  semanal), `cargar_ngs` (passing/receiving/rushing), `cargar_snaps`,
  `cargar_lesiones`, `cargar_qbr` (Total QBR de ESPN, 2006-2025),
  `cargar_stats_equipo` y `cargar_contratos` (OverTheCap, 51.952 filas).
  Nuevo helper `_cargar_global` para ficheros sin `{season}`.
- `estado_datos.py`: semáforo de fuentes — qué hay publicado, hasta qué semana
  llega y cuánto retraso lleva. Deriva la semana de participation desde el
  `nflverse_game_id`, que no tiene columna `week`. Lanzar cada martes.

**Otros huecos detectados, no atacados todavía:** el PBP tiene 372 columnas y
el proyecto usa 122. Sin tocar: `vegas_wp`/`vegas_wpa` (win probability con la
línea), `xyac_epa`/`xyac_mean_yardage` (YAC sobre esperado), `cp`,
`air_epa`/`yac_epa` (cuánto de un QB es su brazo y cuánto sus receptores),
`fixed_drive_result` y el bloque `drive_*` completo, y todo lo de equipos
especiales (`punt_attempt`, `kickoff_*`, `field_goal_attempt`, fair catches).

**Archivos:** pbp_loader.py, estado_datos.py (nuevo), CLAUDE.md,
docs/decisiones.md, docs/scripts-catalog.md, CHANGELOG.md

---

## [2026-08-15] — Auditoría de temporada: 8 fallos arreglados antes del arranque 2026

Auditoría de los 59 scripts ejecutándolos uno a uno contra los datos de 2025.
Ninguno fallaba con datos del año pasado; todos los fallos eran trampas que solo
se abrían al empezar la temporada nueva, más dos que llevaban meses mintiendo.

**Lo que engañaba (datos correctos, conclusión falsa):**
- **DatoSemana — la defensa no podía ganar nunca.** `off_epa` y `def_epa_allowed`
  son los MISMOS 32 valores repermutados (en una semana cada equipo juega un
  partido, y toda jugada tiene posteam y defteam): misma mediana, misma MAD,
  mismo max|z|. Verificado en la semana 9 (1.771 vs 1.771). Como el ataque se
  evaluaba primero y el desempate era `>` estricto, las 3 claves defensivas eran
  código muerto — 0 de 18 semanas en 2025. Ahora solo se evalúan las ofensivas y
  el titular se **reencuadra a defensa cuando el outlier es un desastre**:
  8 de 18 semanas dan ahora titular defensivo.
- **DatoSemana — equipos especiales ganaba 6 de 18 semanas** con `min_plays=6`:
  un punt bloqueado sobre 6 jugadas parecía más extremo que un ataque brillante
  sobre 65. Mínimo a 10 + `penalizacion_muestra()` que encoge el z por tamaño de
  muestra. Baja a 5/18, todas con n=11-15 y EPA de -0.7 a -0.95 (historias reales).
- **pbp_loader — caches congelados en silencio.** Si fallaba la consulta a
  `nfldata/games.csv`, `_info_schedules()` devolvía None y aguas abajo
  `necesita` se quedaba en False: se servía el cache viejo **sin imprimir nada**.
  Un único punto de fallo mudo podía hacer publicar datos de hace semanas con el
  PNG saliendo perfecto. Ahora grita (`_aviso_sin_verificar`) y `sello()` estampa
  "datos hasta sem. N" en la imagen.

**Lo que se rompía en la jornada 1:**
- **Manning_bot estaba clavado en 2025** (`SEASON_PRED = 2025`, entrenamiento
  2015-2024). Era el único script con la temporada a fuego. Ahora
  `resolver_temporadas()` la deriva de schedules (2026, entrenando 2015-2025),
  los folds walk-forward se derivan igual, y `manning_bot_model.meta` detecta un
  modelo entrenado con datos viejos y propone reentrenar. Por defecto pronostica
  la **próxima** jornada, no la última jugada (era un backtest disfrazado).
- **Seis scripts morían con traceback de urllib** mientras FTN/participación 2026
  no estuviera publicado (verificado: 404 en las 4 fuentes). Nueva excepción
  `DatosNoDisponibles` + `sys.excepthook` en pbp_loader (cubre los 50 scripts de
  una vez) + `try` específico en los seis con el motivo del visual.
- **Guardia de muestra**: `aviso_muestra()` dentro de `cargar_pbp` avisa si la
  mediana de jugadas por equipo baja de 150. Antes, el jueves de la jornada 1 se
  dibujaba un ranking de 32 equipos a partir de dos, sin decir nada.

**Lo estructural:**
- **`--season` / `--week` / `--raiz`** en todos los scripts vía `cli()` de
  pbp_loader (antes solo los tenía contenders_tracker). Sin argumentos todo
  funciona igual que antes.
- **Fin de las sobrescrituras**: `salida()` archiva en
  `salidas/{año}/w{semana}/{nombre}_w{NN}.png`. Antes solo 4 de ~50 salidas
  llevaban la semana, así que la de la jornada 4 borraba la de la 3 — y las
  previas, sin semana, hacían que la 2ª de un duelo de división pisara la 1ª.
  `server.py` busca ahora también en `salidas/` (rglob).
- **Recorte de logos por tinta en los 40 scripts** (antes solo 4). NYJ pasa de
  renderizar ~23x7 px de tinta a ~45x14 con zoom 0.05; todos los logos quedan
  igualados por área de tinta real.
- Limpieza: `HARD_PENALTY` (código muerto) fuera de 36 scripts,
  `sys.stdout.reconfigure` en Previas/comparador_tes/generar_dia, docstring de
  `generar_caches.py` corregido (documentaba `lab/generar_caches.py`, que no existe).

**Verificación:** 44 scripts ejecutados end-to-end, 0 tracebacks, 45 PNGs.
`--season 2024` y `--week 10` probados sin teclado.

**Archivos:** pbp_loader.py, Manning_bot.py, DatoSemana.py, server.py, .gitignore
y 45 scripts más (codemods de salidas y logos).

**Pendiente (no son fallos, son huecos de catálogo):**
- `clasificacion.py` — no hay clasificación ni cuadro de playoffs. Es el mayor
  motor de contenido de la semana 10 en adelante. Objetivo: semana 6-8
- Fuentes de nflverse sin explotar: `injuries`, `snap_counts`, `nextgen_stats`
- MVPsSemana y MVPsSeason no generan PNG, solo texto por consola
- Manning_bot sigue sin usar `pbp_loader` (cache duplicado `pbp_{año}.parquet`)

---

## [2026-08-14] — Hilo de SEA publicado, terminología de jugadas y hilo nuevo "Deberes 2026"

**Qué se hizo:**
- **Hilo de SEA publicado** (4 tuits). Cinco rondas de reescritura: se descartaron los arranques por dato en bruto (diferencial, sacks, marcador) porque incumplían la regla de no empezar con números; el arranque definitivo entra por la espera de la franquicia, elegido por Luis entre cuatro opciones
- **Luis cazó un error de bulto**: la versión que decía "cumplían 8 de 12, arreglaron 2" daba 10, no 12. Causa: victorias y diferencial se ponen en verde como consecuencia, no como arreglo. El tuit 1 se rehízo sin aritmética de casillas
- **Terminología de jugadas fijada**: "proteger al QB" / "mejorar la protección de pase" / "pass pro" (nunca "proteger al pasador") y "forzar turnovers" (nunca "robar balones"). Corregidos los tres sitios donde aparecía en el mega-hilo (DEN, SF, CIN)
- **Hilo nuevo `docs/hilo_deberes_2026.md`**: 31 tuits con ángulo prescriptivo (qué tiene que hacer cada equipo en 2026) en vez del retrospectivo de julio. Los dos hilos conviven como alternativas
- Los cuatro equipos de 12/12 se funden en un solo tuit (34 → 31), lo que destapó un dato que por separado no se veía: JAX, BUF y LA se fueron de enero por 3, 3 y 4 puntos
- **Verificación previa a escribir**: fallos por equipo con `ct.compute`, más web para KC (6-11, primera ausencia desde 2014, Mahomes lesionado), draft 2026 (LV #1 Mendoza, NYJ #2 David Bailey, ARI #3, TEN #4), Harbaugh a los Giants y traspaso de Garrett. Descartado un resultado de búsqueda con récords de 16 partidos (6-10, 8-8), imposibles en formato de 17
- Se descartó el calendario de publicación diaria (2 tuits/día del 15 al 31-ago): Luis quiere hilo único, no serie repartida

**Archivos modificados:** docs/hilo_deberes_2026.md (nuevo), docs/hilo_formula_campeon.md, docs/decisiones.md, CLAUDE.md, CHANGELOG.md, memoria (terminología)

**Pendiente:**
- Publicar el hilo "Deberes 2026" (31 tuits, listo y verificado) — decidir si con `contenders_s2025_w18.png` en el tuit 1 o en el cierre
- Post de los 32 equipos, y el tracker semanal desde el 9-sep (arranque: Seahawks–Patriots)
- Sigue pendiente el commit de la sesión del 19-07 (~70 archivos) más todo lo de estos tres días

---

## [2026-08-13] — Hilo de SEA arrancado, dato falso publicado y auditoría de todos los posts pendientes

**Qué se hizo:**
- **Tuit 1 de SEA publicado** (versión B, 274 chars). El cierre anterior ("¿Cuánto dura la era? 🔥") se descartó por robótico; el gancho pasa a ser la defensa que vuelve doce años después de la Legion of Boom
- **⚠️ ERROR PUBLICADO en ese tuit 1**: afirma que las cuatro casillas que SEA arregló eran defensivas. Falso — son dos globales (victorias, diferencial), una defensiva (EPA) y una de **ataque** (sacks permitidos = línea ofensiva). Venía heredado del doc y se propagó sin comprobar. Se arregla con el tuit 2, ya redactado
- **Hilo de SEA estructurado**: tuits 2 (matiz de la línea ofensiva, 54→27 sacks), 3 (números de la defensa) y 4 (playoffs sin turnovers) escritos y pendientes de publicar
- **`evolucion_SEA_2024_2025.png` regenerado**: la versión del 19-jul era anterior al cambio de terminología y decía "3a bajada cedida" → ahora "3er down cedido". Números idénticos
- **Auditoría completa de los 34 tuits del mega-hilo + post de los 32 equipos** contra `contenders_tracker.py` y búsqueda web. Tres errores corregidos: DEN (decía "la defensa ya es de anillo" con una casilla roja defensiva), NYG (Harbaugh, 17 → **18** temporadas en Baltimore), IND (atribuía a Daniel Jones un ataque que terminó sin él, Aquiles en la semana 14). El resto pasa la verificación
- **Dos criterios nuevos** en CLAUDE.md: (a) verificación obligatoria en tres frentes antes de proponer un post; (b) el cierre ilusionante no tiene por qué ser pregunta — prohibidas las retóricas de relleno
- Cierres reescritos por el criterio (b): post de los 32 equipos y tuit 34 del mega-hilo
- Memoria: dos feedbacks nuevos, `verificar-posts` y `cierres-posts`

**Archivos modificados:** CLAUDE.md, docs/hilo_formula_campeon.md, docs/decisiones.md, CHANGELOG.md, evolucion_SEA_2024_2025.png

**Actualización 2026-08-14:** Luis **borró** el tuit publicado con el dato falso. El hilo de SEA se rehízo entero (4 tuits, sin tuit de corrección) con un ángulo nuevo y más fiel: de las 4 casillas rojas de 2024, dos eran resultados (victorias, diferencial) y dos causas (EPA defensivo, sacks permitidos) — arreglaron las causas y los resultados cayeron solos. Todos los números reverificados con `ct.compute` y el contexto por web.

**Hilo de SEA PUBLICADO el 2026-08-14** (4 tuits). Cinco rondas de reescritura hasta dar con el tono; lecciones anotadas en el doc: no abrir con números en bruto (el arranque va por la espera de la franquicia), nada de andamiaje ("Empiezo por…", "El remate…"), nada de aritmética de casillas (la versión "cumplían 8, arreglaron 2" daba 10, no 12) y tono hablado con frases cortas.

**Pendiente:**
- Siguen pendientes, espaciados: post de los 32 equipos y mega-hilo de 34 (mejor cerca del 9-sep, arranque de temporada)
- Sigue pendiente el commit de toda la sesión del 19-07 (~70 archivos) más lo de hoy

---

## [2026-07-27] — Hilo de la fórmula publicado + terminología unificada

**Qué se hizo:**
- **Mega-hilo de 34 tuits terminado**: los 27 pendientes reescritos y, tras feedback de Luis, los 34 rehechos en tono profundo (una idea de fondo por tuit en vez de lista de fallos). Contexto narrativo verificado por búsqueda web: playoffs 2025-26, récords finales y draft 2026 (LV #1 Fernando Mendoza, NYJ #2, ARI #3, TEN #4)
- **Terminología unificada**: "3ª/4ª bajada" → **"3er/4º down"** en todo el proyecto (`contenders_tracker.py` incluida la etiqueta del visual, catálogo, decisiones, CHANGELOG, hilo). `contenders_s2025_w18.png` regenerado con la etiqueta nueva
- Nombres de equipos y ciudades **siempre en inglés** (era "Nueva Inglaterra" en el tuit de NE)
- **Hilo original publicado** (9 tuits, 2026-07-27). Durante la publicación se reescribieron 5 tuits:
  - Tuit 1: hook nuevo ("ningún campeón ha sido una sorpresa")
  - Tuits 6-7 refundidos: las 11 temporadas ya no van partidas (parecía que faltaban años); el 7 pasa a ser el dato de magnitud (46/352, 13,1%)
  - Tuit 8: reflexión reescrita, la anterior sonaba artificial
  - Tuit 9: decía "grupo de 3 a 6" (el dato real es 1-5/año) y hablaba del draft en futuro cuando ya se celebró en abril
- Memoria: nueva feedback `terminologia-downs` (downs en inglés + equipos en inglés, corregir sin preguntar)

**Archivos modificados:** docs/hilo_formula_campeon.md, contenders_tracker.py, docs/scripts-catalog.md, docs/decisiones.md, CHANGELOG.md, contenders_s2025_w18.png

**Pendiente:**
- Publicar (espaciados, no seguidos): post de SEA con `evolucion_SEA_2024_2025.png`, post de los 32 equipos con `contenders_s2025_w18.png`, y el mega-hilo de 34
- Decidir imágenes para el resto de posts del mega-hilo
- Sigue pendiente el commit de toda la sesión del 19-07 (~70 archivos sin commitear)

---

## [2026-07-19] — Dashboard blindado, arreglos visuales, script de origen de presión, informe_equipo rediseñado

**Qué se hizo:**
- `server.py`: bug crítico — un proceso zombi ignoraba el modo "Ver análisis" y llenaba la raíz de PNGs; reiniciado y blindado (`png` ahora es `false` por defecto, no `true`). 69 PNGs sueltos movidos a `salidas/` (gitignored)
- `resumen_partido.py`: rediseño completo — boxscore de 8 filas + curva de Win Probability con Top-3 por \|WPA\|, y segundo PNG "Claves del partido" (top-4 desviaciones de cada equipo vs su norma de temporada, con contexto del rival)
- `ranking_rbs.py` (+ mismo fix portado a wrs/tes): logos recortados por tinta real (no por lienzo) y etiquetas con halo + línea guía — arregla logos enanos/invisibles y nombres perdidos
- Nuevo `oline_presion_origen.py`: heatmap 32 equipos y diagrama de campo con el origen de la presión cedida (interior/exterior/blitz LB/blitz DB), vía atribución de sacks+QB hits y posiciones del roster
- `ataque_vs_personal_ofensivo.py` renombrado a `ataque_por_personal_ofensivo.py` (el "vs" era engañoso)
- `informe_equipo.py` rediseñado como "team card": KPIs con ranking, facetas como puntos en pista de ranking 1→32, fortalezas/debilidades autogeneradas, identidad (uso vs media NFL); fix de denominador en "Ve zona"/"Juega zona" (ahora sobre jugadas con cobertura charteada, no todas las jugadas)
- `informe_equipo.py`: sección "Identidad" (ataque) sustituye "Ve zona"/"Sufre presión" (describían al rival, no al equipo) por Shotgun%/Bajo centro%/No-huddle%/Play-action% — decisiones propias de play-calling (shotgun y no_huddle del PBP; is_play_action de FTN charting, 2022+)
- `informe_equipo.py`: cards "Donde domina"/"Donde sufre" pasan de altura fija a dinámica según nº de items; el espacio libre se lo lleva "Identidad", que ya no capa personal a 3 paquetes
- `docs/pff-wishlist.md` (nuevo): 8 datos + 1 script (`dline_presion_origen.py`) pendientes para cuando se suscriba a PFF+ (decisión: anual, finales de agosto 2026)
- Memoria: nueva feedback `atajos_datos` — no presentar proxies como el dato real cuando algo no existe en abierto
- `discriminacion_total.py` eliminado (su fórmula ya vive en `contenders_tracker.py`); limpiadas sus referencias en catálogo, `generar_caches.py` y wishlist
- Verificado empíricamente que los 12 requisitos del tracker SÍ discriminan en conjunto (no solo por construcción): 46/352 equipo-temporada 2015-2025 los cumplen todos (13.1% de la liga), con los 11 campeones siempre dentro
- Nuevo `evolucion_contender.py`: compara un equipo entre 2 temporadas contra los 12 requisitos (reutilizable para cualquier equipo/año)
- `docs/hilo_formula_campeon.md`: hilo original recalibrado con umbrales/conteos actuales + 3 posts de seguimiento añadidos (evolución SEA 2024→2025, resumen 32 equipos, mega-hilo de 34 tuits equipo a equipo) — el mega-hilo quedó pendiente de reescribir en tono menos robótico (feedback del usuario, con muestra de 7 tuits ya aprobada como dirección)

**Archivos modificados:** server.py, resumen_partido.py, ranking_rbs.py, ranking_wrs.py, ranking_tes.py, oline_presion_origen.py (nuevo), ataque_por_personal_ofensivo.py (renombrado), informe_equipo.py, evolucion_contender.py (nuevo), galeria.html, docs/scripts-catalog.md, docs/pff-wishlist.md (nuevo), docs/hilo_formula_campeon.md, docs/decisiones.md, .gitignore, discriminacion_total.py (eliminado)

**Pendiente:**
- `dline_presion_origen.py` (espejo defensivo del origen de presión) — esperando suscripción PFF de agosto 2026
- Reescribir los 34 tuits del "post de seguimiento 3" en tono humano (la muestra de 7 ya está aprobada en el hilo de conversación, falta aplicarla a los 27 restantes)
- Decidir imágenes para acompañar cada post del hilo

---

## [2026-06-12] — Auditoría completa: pbp_loader.py + 43 scripts migrados

**Qué se hizo:**
- `pbp_loader.py` (nuevo, única excepción a "scripts independientes"): carga compartida con cache local (`pbp_full_{yr}.parquet`, parquet oficial de nflverse), filtro REG, auto-detección de temporada y re-descarga solo cuando hay jornadas nuevas. También `cargar_stats()` y `cargar_participation()`
- **43 scripts migrados al loader**. Tres bugs/problemas sistémicos resueltos de golpe:
  1. ~45 scripts mezclaban playoffs en stats "de temporada" (no filtraban season_type) — ahora REG por defecto
  2. ~45 scripts re-descargaban 90MB de PBP en cada ejecución — ahora cache local compartido
  3. `SEASON=2025` hardcodeado en 43 scripts — ahora `SEASON=None` auto-detecta (lista para 2026 sin tocar nada)
- Scripts de partido/semana concreta (resumen_partido, Previas, DatoSemana, MVPsSemana, power_rankings) usan `solo_reg=False`: mantienen acceso a playoffs vía su propio filtro de semana
- `contenders_tracker.py`: blindado el manejo de booleanos numpy (`is True` → `pd.isna`/`bool()`)
- `server.py`: solo ejecuta `.py` que vivan en la carpeta del proyecto (validación de ruta)
- `galeria.html`: añadida entrada para contenders_tracker
- CLAUDE.md: nota de participación corregida (FTN 2016-2025 con rutas/presión/coberturas) y convención de carga documentada
- Verificado: 58 scripts compilan; smoke tests OK (qb_overview, coberturas, comparador_wrs end-to-end con cache)

**Pendiente (huecos detectados en la auditoría):** special_teams.py, air_yards.py, arbol_rutas.py, presion_blitz.py, man_vs_zone.py, penalizaciones.py, simulador_playoffs.py

---

## [2026-06-11] — contenders_tracker.py autosuficiente + recalibración de umbrales

**Qué se hizo:**
- `contenders.py` borrado: superado por `contenders_tracker.py --season X` (mismos criterios, pero recalibrados y con datos frescos; el viejo daba resultados inconsistentes)
- `contenders_tracker.py` ya no depende de caches congelados: descarga schedules frescos en cada ejecución (y actualiza `schedules.parquet`) y el PBP de la temporada solo cuando detecta jornadas nuevas (cache `pbp_contenders_{season}.parquet`). Listo para uso semanal en 2026: `python contenders_tracker.py` y sale el PNG al día
- Métricas calculadas directamente del PBP replicando las definiciones de calibración (game logs de Manning_bot); YPA y sacks ahora estilo oficial (intentos sin sacks, yardas de pases completados)
- **Bug de datos corregido**: las filas QB 2025 de `player_stats` tenían `sacks=0.0` (aproximación), así que el criterio "sacks permitidos" lo pasaba todo el mundo gratis en 2025, y el YPA estaba deflactado por incluir yardas de sack
- Umbrales recalibrados al peor campeón con las nuevas definiciones (solo se relajaron los 5 que fallaban por centésimas): def_epa ≤-0.015, pts cedidos pace ≤378, 3er down ≤0.415, sacks pace ≤45.5, yds/jugada ≤5.85 (los dos últimos ya estaban anotados como pendientes en el CHANGELOG del 29-05 pero nunca llegaron al tracker)
- Re-verificado el patrón 11/11: el campeón siempre entre los contenders (2015-2025), embudos de 1-5 equipos/año

**⚠️ Cambia un resultado publicable:** los contenders finales 2025 ahora son **BUF, JAX, LA, SEA** (antes LA, NE, SEA — NE falla sacks permitidos con datos reales, BUF/JAX entran al corregirse su YPA). Revisar los conteos del embudo en `docs/hilo_formula_campeon.md` antes de publicar el hilo.

---

## [2026-06-11] — Limpieza pipeline fórmula del campeón: lab/ graduado a raíz

**Qué se hizo:**
- `lab/` eliminada: `discriminacion_total.py`, `contenders.py` y `contenders_tracker.py` promovidos a la raíz (son la versión definitiva de la fórmula)
- `generar_caches.py` (nuevo): consolida en un solo script parametrizado por año la generación de los 6 caches (fullmetrics, epa_type, situational, explosive, rz_def, newmetrics) + QBs en player_stats. Uso: `python generar_caches.py 2026 [--force]`
- Validado contra los caches existentes: 6/6 idénticos en 2025; en 2015 todo idéntico salvo `trailing_success_rate` de situational (ver nota)
- Borrados 5 scripts v1 superados: `formula_campeon.py`, `formula_tracker.py`, `sintesis_formula.py`, `nuevas_metricas.py`, `generar_caches_2025.py`
- `docs/scripts-catalog.md` actualizado; convención lab/ documentada en CLAUDE.md

**Notas:**
- ⚠️ El hilo `docs/hilo_formula_campeon.md` referenciaba los frames de `formula_campeon.py` como imágenes candidatas — al borrarlo, el hilo necesita imágenes nuevas
- ⚠️ Inconsistencia histórica detectada (preexistente): `situational_2015-2024` se generaron con una definición de "trailing" distinta a la de 2025 (`trailing_success_rate` difiere hasta 0.045). Si se usa esa métrica entre años, conviene regenerar 2015-2024 con `generar_caches.py --force`

---

## [2026-05-29] — Hilo X fórmula del campeón finalizado con embudo real

**Qué se hizo:**
- `docs/hilo_formula_campeon.md`: hilo reescrito con embudo real verificado con datos
- 12 criterios validados contra los 11 campeones (2015-2025) — 11/11 OK
- Umbrales ajustados por precisión flotante: yds/play ≤5.85 (NE 2018=5.7809), def_3rd ≤0.415 (LA 2021=0.4104)
- Embudo calculado con datos reales: 32 → ~7 → ~6 → ~5 → ~4 equipos/año
- Año a año (C4) confirmado: campeón siempre en el grupo en las 11 temporadas
- Posts con conteos correctos por año: 2016 solo 1 equipo (NE), 2022 hasta 6 (KC)
- Estado: listo para publicar, pendiente decidir imágenes a adjuntar

**Archivos modificados:**
- `docs/hilo_formula_campeon.md` — reescrito completo, tabla de criterios añadida

---

## [2026-05-29] — Contenders tracker depurado + borrador hilo X

**Qué se hizo:**
- `lab/contenders_tracker.py`: corregidos 2 bugs críticos
  - `drive_score_rate` umbral 0.35 → 0.3461 (valor exacto del peor campeón, PHI 2024)
  - Filtro por semana en player_stats vaciaba el dataset cuando la columna `week` es todo NaN (datos de temporada); corregido con `notna().any()`
- Verificado que PHI 2024 y SEA 2025 aparecen como 12/12 CONTENDER al final de sus respectivas temporadas
- Confirmado patrón histórico 11/11: el campeón siempre entre los contenders (2015-2025)
- Generado `contenders_s2025_w18.png` en raíz — 3 contenders finales 2025: LA, NE, SEA
- Borrador hilo X "La fórmula del campeón" — 9 tweets — guardado en `docs/hilo_formula_campeon.md`

**Archivos modificados/creados:**
- `lab/contenders_tracker.py` — 2 bugs corregidos
- `docs/hilo_formula_campeon.md` — nuevo, borrador hilo completo
- `contenders_s2025_w18.png` — nuevo PNG en raíz

**Pendiente:**
- Diseñar imágenes para el hilo (¿frames de formula_campeon.py o nuevos?)
- Revisar conteos de caracteres definitivos antes de publicar

---

## [2026-05-20] — Hilo "La fórmula del campeón" + tracker semanal

**Qué se hizo:**
- Análisis completo de 11 campeones (2015-2025) incluyendo SEA 2025 — fórmula encontrada
- Exploradas 20+ métricas (EPA, scoring, yardas, penalidades, QB hits, YAC, success rate DEF, etc.)
- `formula_campeon.py`: ampliado a 11 campeones, umbrales corregidos (ypd 6.1→5.3, pts 174→400), def_sacks eliminado (0/11), añadidas 3 métricas al heatmap (success_rate_def, qb_hits_pg, yac_avg)
- `formula_post.py` (nuevo): genera `hilo_09_formula.png` — post visual con las 2 reglas universales
- `formula_tracker.py` (nuevo): tracker semanal PNG de equipos que cumplen la fórmula, auto-detecta última semana
- `nuevas_metricas.py` + `sintesis_formula.py` (nuevos): scripts de análisis auxiliares
- `generar_caches_2025.py` (nuevo): genera los 5 caches de temporada 2025 desde PBP completo
- Portada corregida: grid dinámico para N campeones (antes hardcodeado a 10, SEA quedaba fuera)
- 7 PNGs obsoletos borrados de la raíz

**Archivos modificados/creados:**
- `formula_campeon.py` — ampliado y corregido
- `formula_post.py`, `formula_tracker.py`, `nuevas_metricas.py`, `sintesis_formula.py`, `generar_caches_2025.py` — nuevos
- `pbp_cache/newmetrics_{2015-2025}.parquet` — nuevos caches de métricas adicionales

**Pendiente:** redactar tweets del hilo (hilo_01→hilo_09), publicar

---

## [2026-05-18] — Hilo Shanahan: revisión visual completa + tweets

**Qué se hizo:**
- Revisión gráfica a gráfica de las 12 visualizaciones del hilo Shanahan
- `hilo_mcvay_shanahan_v5.py`: reemplazado hilo_12 de barras por scatter de TEs en R1 con nombres y pick en eje Y
- Descartado hilo_12 de total TEs (pre/post 2017 idéntico: 14.3 vs 14.4) — R1 cuenta la historia (3 vs 11)
- Redactados 12 tweets + intro + cierre para el hilo completo
- Ajustes visuales menores: posición "Shanahan llega" en hilo_12, eje Y invertido en scatter

**Archivos modificados:**
- `hilo_mcvay_shanahan_v5.py` — hilo_12 reescrito como scatter
- `hilo_12_draft_te_r1.png` — regenerado

**Pendiente:** ninguno — hilo listo para publicar

---

## [2026-05-16] — Nuevos scripts y mejoras visuales

**Qué se hizo:**
- `matchup_intel.py`: añadido segundo PNG de resumen con top 3 armas del atacante y top 3 fortalezas del defensor
- `qb_overview.py`: nuevo script — scatter EPA/play vs CPOE para todos los QBs, color = EPA bajo presión, tamaño = volumen de intentos
- `game_script.py`: nuevo script — cómo rinde un equipo según el contexto del marcador (EPA, pass rate vs liga, 3er down, 4th down); modo equipo 2×2 + scatter 32 equipos
- Corrección de colores en `qb_overview.py`: normalización por rango real en lugar de centrado en 0
- Múltiples fixes de solapamiento visual en `game_script.py` (etiquetas, leyenda 4th down, márgenes ylim)

**Archivos modificados:**
- `matchup_intel.py` — bloque resumen top 3 al final
- `qb_overview.py` — nuevo
- `game_script.py` — nuevo
- `galeria.html` — entradas añadidas para los 3 scripts anteriores; corregido PNG de QBsTotalEPA

**Pendiente:**
- `air_yards.py` — distribución air yards vs YAC por receptor
- `special_teams.py` — FG%, EPA kicker/punter/retornos

---
