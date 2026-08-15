# Decisiones técnicas — NFL 2025

---

## [2026-08-15] — `pbp_participation` no se actualiza en temporada: hay que vigilarlo

**Decisión:** Se añade `estado_datos.py`, un semáforo que reporta qué fuentes
hay publicadas para una temporada, hasta qué semana llegan y cuánto retraso
llevan respecto a la liga. Se lanza cada martes antes de producir.

**Motivo:** Al auditar la cadencia real de nflverse (leyendo los `cron` de
`nflverse-pfr`, `ngs-data`, `nflverse-rosters` y `nflverse-ftn`, y contando
ejecuciones) apareció que **`update_participation.yaml` no tiene `schedule`: es
`workflow_dispatch`**. En todo el historial disponible hay 8 ejecuciones y una
sola durante la temporada 2025 — el 10-feb-2026, ya acabada, que rellenó las 22
semanas de golpe. De `pbp_participation` salen cobertura, personal, presión y
rutas, que alimentan 13 scripts. `ftn_charting` sí corre cada 6 h pero **no
contiene esas columnas** (comprobado: 29 columnas, ninguna de cobertura ni
personal), así que no sirve de sustituto.

El fallo que esto puede producir es del tipo peligroso: el gráfico sale
perfecto y los datos son de la temporada pasada. Por eso la respuesta no es
solo un `try/except` —que ya existe y cubre el caso de "no existe"— sino una
comprobación explícita de **hasta qué semana llega cada fuente**.

**Alternativas descartadas:**
- Migrar los 13 scripts a `ftn_charting` → no tiene las columnas necesarias
- Confiar en el `try/except` de DatosNoDisponibles → solo cubre el caso ruidoso
  (no publicado), no el silencioso (publicado pero congelado)

**Pendiente:** confirmar en la semana 2 de sep-2026 si `pbp_participation_2026`
aparece. Si no, cobertura y personal quedan sin fuente gratuita y PFF pasa de
lujo a plan B.

---

## [2026-08-15] — `pfr_advstats` cubre media wishlist de PFF, gratis

**Decisión:** Se añaden 7 cargadores a `pbp_loader` para fuentes que el
proyecto no usaba: `cargar_pfr`, `cargar_ngs`, `cargar_snaps`,
`cargar_lesiones`, `cargar_qbr`, `cargar_stats_equipo` y `cargar_contratos`.

**Motivo:** La auditoría de agosto fue centrada en el código y no revisó el
universo de datos disponible; Luis lo señaló. Al hacerlo apareció
`pfr_advstats` (190 assets, estadísticas avanzadas de Pro Football Reference,
cada 6 h en temporada) que cubre **los items 3 y 6 de `docs/pff-wishlist.md`
exactamente como estaban especificados**: placajes fallados (`m_tkl`,
`m_tkl_percent`) y cobertura por defensor (`tgt`, `cmp_percent`, `yds_tgt`,
`rat`, `dadot`). También trae presiones por jugador (`prss`, `hrry`, `qbkd`),
que la wishlist daba por imposible sin pagar.

Siguen siendo propietarios de PFF los items 1 y 8 —presión cedida por liniero
concreto y grades de bloqueo por hueco—, que son los que motivaron la lista.

**Alcance:** No cancela la suscripción; recomienda retrasarla para llegar a
ella sabiendo qué falta de verdad. Además apareció QBR semanal de ESPN
(`espn_data`, 2006-2025) y contratos de OverTheCap (51.952 filas con `apy`,
`apy_cap_pct` y enlace por `gsis_id`), ninguno usado.

**Lección de método:** «qué falta» no se responde inventariando scripts. El
catálogo estaba bien construido sobre las 4 fuentes que usaba; el hueco estaba
en las 21 que no. Una auditoría de cobertura tiene que empezar por los datos
disponibles, no por el código existente.

---

## [2026-08-15] — Las métricas defensivas de DatoSemana eran código muerto: reencuadre en vez de evaluación

**Decisión:** `DatoSemana.py` evalúa solo métricas ofensivas y de equipos
especiales. Las tres defensivas se eliminan de la lista de candidatas y pasan a
`ESPEJO_DEF`, que se usa para **reencuadrar** el titular cuando el outlier es
negativo: un ataque hundido se cuenta como la gran actuación de la defensa rival.

**Motivo:** No era una preferencia editorial sino una imposibilidad matemática.
En una semana cada equipo juega un partido y toda jugada tiene `posteam` y
`defteam`, así que los 32 valores de `off_epa` y los de `def_epa_allowed` son el
mismo conjunto repermutado: idéntica mediana, idéntica MAD, idéntico max|z|
(verificado en la semana 9: 1.771 y 1.771). Con el ataque evaluado primero y
desempate `>` estricto, la defensa no podía ganar jamás — 0 de 18 semanas de
2025. Con el reencuadre, 8 de 18 dan titular defensivo.

**Alternativas descartadas:**
- Invertir el orden de la lista → simetría al revés, ahora nunca ganaría el ataque
- Desempatar al azar → el resultado dejaría de ser reproducible semana a semana

---

## [2026-08-15] — El z-score se encoge por tamaño de muestra

**Decisión:** `penalizacion_muestra(n, k=15)` multiplica el z robusto por
`sqrt(n/(n+k))`, y el mínimo de jugadas de equipos especiales sube de 6 a 10.

**Motivo:** Con `min_plays=6`, equipos especiales ganaba 6 de 18 semanas de 2025.
Una media de EPA sobre 6-10 jugadas la decide un punt bloqueado; sobre 65 jugadas
ofensivas, no. El z bruto no distingue entre las dos y premiaba sistemáticamente
al estimador más ruidoso. Tras el cambio ST gana 5/18, pero todas con n=11-15 y
EPA de -0.7 a -0.95: días de teams genuinamente históricos, que sí son noticia.

**Alternativas descartadas:**
- Penalizar con `sqrt(n/n_referencia)` → demasiado agresivo, ST no ganaría nunca
  (el mismo error de simetría que teníamos con la defensa, en el otro sentido)
- Excluir equipos especiales → hay semanas en que es la mejor historia

---

## [2026-08-15] — Un cache servido sin verificar tiene que gritar

**Decisión:** Cuando `_info_schedules()` falla, `pbp_loader` imprime un aviso en
bloque y `sello()` estampa "frescura sin verificar" en el pie del PNG. Se añade
`DatosNoDisponibles` + un `sys.excepthook` para que un dataset que aún no existe
no salga como traceback de `urllib`.

**Motivo:** El fallo más peligroso del proyecto era silencioso: si la consulta a
`nfldata/games.csv` fallaba, `necesita` se quedaba en `False` y se servía el
cache viejo sin decir nada. El gráfico salía perfecto —logos, colores, marca de
agua— con datos de hace semanas. Todos los demás fallos son ruidosos; este era el
único capaz de llevar a publicar un dato falso, que es justo lo que la regla de
verificación en tres frentes intenta evitar.

**Alcance:** El aviso por consola es la red de seguridad; el sello en la imagen es
la que aguanta, porque sobrevive a que el aviso pase desapercibido.

---

## [2026-08-15] — La semana va en el nombre del fichero, y es la semana de los DATOS

**Decisión:** `salida()` archiva en `salidas/{año}/w{NN}/{nombre}_w{NN}.png`. La
semana por defecto no la elige el usuario: es `ultima_semana()`, hasta donde
llegan los datos. Con `--raiz` se conserva el comportamiento antiguo.

**Motivo:** Solo 4 de ~50 salidas llevaban la semana, así que cada jornada
borraba la anterior y no quedaba archivo — justo lo que hace falta en diciembre
para contar la evolución de la temporada. Además las previas no llevaban semana,
y los duelos de división se juegan dos veces: la segunda pisaba la primera.
Usar la semana de los datos (y no una pedida por teclado) hace que el nombre del
fichero sea una afirmación verificable sobre su contenido, en la misma línea que
el sello del pie.

**Alternativas descartadas:**
- Dejar los PNG en la raíz con sufijo → resuelve el archivo pero no el desorden;
  ya había 32 PNGs versionados y cada jornada tocaría ~50 binarios en git
- Timestamp en vez de semana → ordena, pero no dice nada sobre los datos

---

## [2026-08-15] — Manning_bot deriva la temporada del calendario, no de una constante

**Decisión:** `resolver_temporadas()` toma la temporada a predecir como la máxima
de `schedules` y entrena con todas las anteriores con resultados. Los folds
walk-forward se derivan igual. `manning_bot_model.meta` guarda la última
temporada usada para detectar un modelo caducado.

**Motivo:** `SEASON_PRED = 2025` convertía el predictor en un backtest en cuanto
pasaba un verano, sin avisar: habría dado pronósticos de partidos jugados hacía un
año con aspecto de funcionar bien. Usar el máximo de schedules y no
`temporada_actual()` es deliberado: en agosto la primera devuelve 2026 (el
calendario ya está publicado) y la segunda todavía 2025.

**Coste aceptado:** hay que reentrenar una vez para incorporar 2025; el aviso de
modelo caducado lo recuerda al arrancar.

---

## [2026-07-19] — discriminacion_total.py eliminado del repo

**Decisión:** Borrar `discriminacion_total.py`. Su conclusión (la fórmula de
2 reglas: Wins≥11 + Top-7 anotados O Top-7 cedidos) ya vive documentada aquí
mismo y ya no se ejecuta como script — `contenders_tracker.py` es el que se
usa activamente para el seguimiento semanal.

**Motivo:** Petición directa del usuario ("bórrala, nos quedamos con el
otro"). El script no tenía ejecuciones programadas ni el dashboard lo exponía
(no estaba en galeria.html); su único consumidor real era un comentario en el
docstring de `generar_caches.py`, que además ya mencionaba un `contenders.py`
inexistente — estaba desfasado.

**Alternativas descartadas:**
- Mantenerlo en `lab/` "por si acaso" → rechazado, no hay lab/ en este
  proyecto (se eliminó en una limpieza anterior, ver entrada 2026-06-11) y
  reintroducirla solo para un script parking no aporta valor

**Coste aceptado:** si algún día se quiere re-validar la fórmula desde cero
contra más métricas (p.ej. al incorporar datos de PFF), habrá que
reconstruir ese análisis — no quedó guardado en ningún sitio ejecutable.
Ver `docs/pff-wishlist.md` item 2.

---

## [2026-07-19] — Presión por línea ofensiva: origen del rusher, no proxy con/sin

**Decisión:** `oline_presion_origen.py` atribuye la presión cedida al ORIGEN del
rusher (interior/exterior/blitz LB/blitz DB vía sacks+QB hits del PBP y
posiciones del roster), pero NO muestra qué liniero concreto (LT vs RT) fue
batido. Se probó un proxy "presión del equipo con/sin cada titular en campo"
y se descartó a petición del usuario.

**Motivo:** El usuario rechazó explícitamente el proxy: "no es lo que quería...
si no se puede pagar dímelo". La presión por liniero individual es charting
propietario (PFF Premium Stats y equivalentes) — no existe en datos públicos
reproducibles. Precio verificado: PFF+ $9.99/mes o $79.99/año.

**Alternativas descartadas:**
- Proxy con/sin titular en campo (splits de equipo) → implementado y luego
  retirado del diagrama por decisión del usuario, no es lo mismo que la
  atribución real por jugador
- Scraping de ESPN Pass Block Win Rate → solo publica top-10 por artículo, no
  dataset completo ni reproducible semana a semana

**Ver también:** `docs/pff-wishlist.md` — script `dline_presion_origen.py`
(espejo defensivo) pospuesto deliberadamente hasta tener datos PFF completos,
en vez de construir una v1 a medias con solo datos públicos.

---

## [2026-07-19] — Denominador de "% uso" en informe_equipo.py

**Decisión:** `snap_pct()` calcula el % sobre las jugadas CON dato en esa
faceta (p. ej. pases con cobertura charteada), no sobre el total de jugadas
del equipo (que incluye carreras sin charting de cobertura).

**Motivo:** Con el total de jugadas como denominador, "Ve zona" salía 41%
para un equipo que en realidad ve zona el ~70% de sus pases — dato diluido y
que no cuadraba con el desglose de coberturas de la misma página. El usuario
detectó la inconsistencia al leer el gráfico.

---

## [2026-07-19] — informe_equipo.py: "Identidad" solo con decisiones propias

**Decisión:** En el lado ofensivo, "Ve zona"/"Sufre presión" se sustituyen por
Shotgun%/Bajo centro%/No-huddle%/Play-action%. En el lado defensivo, "Juega
zona"/"Genera presión" NO se tocan.

**Motivo:** "Ve zona" y "Sufre presión" describen lo que el RIVAL le hace al
ataque, no una decisión propia — encajaban mal en una sección llamada
"Identidad" y además duplicaban el desglose de Hombre/Zona y Presión de la
columna izquierda de la misma página. "Juega zona"/"Genera presión" en
defensa SÍ son decisiones propias del equipo (qué cobertura eligen jugar, si
generan presión), así que ahí no había el mismo problema. Confirmado con el
usuario vía pregunta directa antes de tocar el código.

**Alternativas descartadas:**
- Mantener ve zona/presión y añadir las nuevas encima → rechazado, eran
  redundantes con otra sección de la misma página
- Enseñar ambas versiones para decidir mirando → el usuario prefirió ir
  directo a la opción recomendada

---

## [2026-05-20] — Fórmula del campeón: métricas y umbrales definitivos

**Decisión:** La fórmula se reduce a 2 reglas universales (11/11 campeones, 0 excepciones):
1. Wins ≥ 11 en temporada regular
2. Top-7 en puntos anotados OR Top-7 en puntos cedidos (ambos ≤ #20)

**Motivo:** Análisis exhaustivo de 20+ métricas sobre 11 temporadas (2015-2025). Estas 2 son las únicas con 100% de cumplimiento y buena discriminación real (~26% de la liga califica, no 50%+). Métricas como EPA def < 0, TO diff > 0 o sack rate son universales pero triviales (≥49% de la liga las cumple). Sacks DEF > 50 completamente refutada (0/11 campeones).

**Alternativas descartadas:**
- Umbral único de EPA defensivo < -0.05 → solo 7/11 campeones
- Regla de sacks > 50 (propuesta por herramienta externa "SumerBrain") → 0/11, basada en solo 4 temporadas
- Success rate DEF como requisito único → 10/11 pero 67% de la liga también califica
- Métricas PFF (missed tackles, pressure rate real) → datos propietarios, no disponibles en nflverse

---

## [2026-05-20] — Tracker semanal: fuente de datos sin descargar PBP

**Decisión:** El tracker (`formula_tracker.py`) usa exclusivamente `schedules.parquet` y `game_logs_all.parquet` — sin descargar PBP completo cada semana.

**Motivo:** El PBP completo pesa ~50MB por temporada y tarda 30-60s en descargar. Las métricas de la fórmula (wins, pts_scored_rank, pts_allowed_rank, EPA) están todas disponibles en los dos parquets ya cacheados, que nflverse actualiza semanalmente de forma ligera.

**Alternativas descartadas:** Descarga semanal de PBP completo para calcular yds/play DEF con más precisión → latencia inaceptable para uso rutinario semanal.

---

## [2026-06-11] — Tracker autosuficiente con umbrales recalibrados (sustituye la decisión del 20-05)

**Decisión:** `contenders_tracker.py` descarga sus propios datos frescos (schedules + PBP con cache incremental por semana) y calcula las 12 métricas directamente del PBP. Umbrales recalibrados al peor campeón con estas definiciones (def_epa ≤-0.015, pts cedidos pace ≤378, 3er down ≤0.415, sacks pace ≤45.5, yds/jugada ≤5.85). Re-verificado 11/11 campeones 2015-2025.

**Motivo:** Los caches congelados (game_logs, player_stats, newmetrics) no se actualizaban nunca → el tracker habría dicho "sin datos" en 2026. Además las filas QB 2025 de player_stats tenían `sacks=0.0`, lo que invalidaba el criterio de sacks. La latencia ya no es problema: el PBP se cachea y solo se re-descarga con jornadas nuevas.

**Alternativas descartadas:** Script semanal separado de actualización de caches (`actualizar_semana.py`) → el usuario prefirió apilar todo en el script original. Mantener umbrales antiguos → 6/11 campeones fallaban por centésimas con las definiciones corregidas.

---

## [2026-06-12] — pbp_loader.py: única excepción a "scripts independientes"

**Decisión:** Módulo común `pbp_loader.py` (`cargar_pbp/cargar_stats/cargar_participation`) que todos los scripts importan. Cache local del parquet oficial de nflverse, filtro REG por defecto, auto-detección de temporada.

**Motivo:** 45 scripts duplicaban el mismo bloque de carga con 3 defectos sistémicos (playoffs mezclados en stats de temporada, re-descarga de 90MB por ejecución, año hardcodeado). Arreglarlo en un solo sitio frente a mantener 45 copias.

**Alternativas descartadas:** Copiar el bloque corregido en cada script (mantenía la independencia total pero multiplicaba el mantenimiento ×45). Los scripts de partido/semana concreta usan `solo_reg=False` para no perder acceso a playoffs.

---

## [2026-07-27] — Terminología del proyecto: "down" y equipos en inglés

**Decisión:** En todo texto de cara al público (posts, etiquetas de visuales, docs) se escribe **"3er down" / "4º down"**, nunca "3ª bajada"; y los equipos y sus ciudades van **en inglés** (New England, no "Nueva Inglaterra"). Aplicado también a las etiquetas de `contenders_tracker.py`, que obligó a regenerar `contenders_s2025_w18.png`.

**Motivo:** Es la terminología que usa la audiencia hispanohablante de NFL. Luis lo corrigió dos veces (jul-2026), así que queda fijado aquí y en CLAUDE.md para no volver a discutirlo.

**Alternativas descartadas:** Traducir todo al castellano ("bajada", "Nueva Inglaterra") → suena ajeno al aficionado real. Mezclar criterios según el contexto → inconsistencia entre visual y post.

---

## [2026-08-13] — Cierre de los posts: nada de preguntas retóricas de relleno

**Decisión:** El cierre ilusionante **no tiene por qué ser una pregunta**. Quedan prohibidas las preguntas retóricas genéricas que valdrían para cualquier equipo ("¿Cuánto dura la era?", "¿Quién cierra la grieta?", "¿Quién da el salto en 2026?"). Si el cierre no aporta información ni opinión, el post termina en el dato o en una frase corta y afirmativa.

**Motivo:** Luis rechazó el cierre del post de SEA por sonar robótico. La plantilla [historia → dato → pregunta → hashtags] estaba produciendo el mismo final intercambiable en todos los posts, y el patrón se nota cuando se publican seguidos.

**Alcance:** Aplicado al post de SEA (versión B, publicable) y reescritos los cierres del post de los 32 equipos y del tuit 34 del mega-hilo, que arrastraban el mismo vicio. Fijado en CLAUDE.md.

**Alternativas descartadas:** Prohibir las preguntas del todo → una pregunta genuina, con contenido propio, sigue siendo un buen cierre (el tuit 9 del hilo original funciona). Lo que se prohíbe es la pregunta de relleno.

---

## [2026-08-13] — El doc del hilo no es fuente de verdad: verificación en tres frentes

**Decisión:** Ningún post se propone para publicar sin pasar tres verificaciones: (1) **los números**, ejecutando el script que genera el visual (`contenders_tracker.py`), nunca leyendo el doc ni el PNG a ojo; (2) **la atribución ataque/defensa** de cada métrica contra su categoría real — sacks permitidos es línea ofensiva = ATAQUE, y victorias y diferencial de puntos son globales, no defensivas; (3) **nombres, récords, traspasos y resultados**, con búsqueda web.

**Motivo:** Se publicó un tuit de SEA afirmando que las cuatro casillas arregladas eran defensivas cuando solo una lo era. El error estaba en el borrador de `docs/hilo_formula_campeon.md` desde julio y se propagó al reescribir el post porque se tomó el doc como fuente. Un dato falso publicado no se puede deshacer. La auditoría posterior de los 34 tuits pendientes destapó tres errores más (DEN, NYG, IND), lo que confirma que no era un caso aislado sino un fallo de método.

**Alcance:** Fijado en CLAUDE.md como regla dura y en memoria como feedback `verificar-posts`. `docs/hilo_formula_campeon.md` queda explícitamente degradado a **borrador**, no a referencia.

**Alternativas descartadas:** Verificar solo los posts nuevos → los errores vivían justamente en el material ya escrito y dado por bueno. Confiar en el PNG como fuente → la etiqueta correcta puede convivir con una lectura equivocada de la categoría de la métrica, que es exactamente lo que pasó.

---

## [2026-08-14] — Victorias y diferencial son consecuencia, no deberes

**Decisión:** En cualquier texto sobre la fórmula del campeón, las casillas de **Victorias** y **Diferencial de puntos** no se cuentan ni se presentan como tareas a arreglar: son el resultado de arreglar las causas (EPA defensivo, protección, pase, etc.). En el hilo "Deberes 2026" solo se listan como deberes las métricas causales, y victorias/diferencial aparecen únicamente como resultado.

**Motivo:** La formulación "cumplían 8 de 12 y arreglaron 2" no cuadraba — daba 10, no 12 — porque las otras dos casillas se pusieron verdes solas. Luis lo detectó al leer el borrador. El problema no era de redacción sino conceptual: tratar consecuencias como si fueran palancas.

**Alcance:** Aplicado al hilo de SEA publicado y a los 31 tuits de `docs/hilo_deberes_2026.md`. La excepción documentada es Philadelphia, cuyo único fallo *es* el diferencial: su tuit dice "ninguno técnico".

**Alternativas descartadas:** Contar las cuatro casillas como deberes → es lo que producía la aritmética imposible. Omitir el conteo por completo → se pierde el gancho de "cuántos deberes tiene cada equipo", que es la columna vertebral del hilo.

---

## [2026-08-14] — Dos hilos con el mismo dato y distinto ángulo

**Decisión:** Mantener **dos hilos completos y verificados** sobre la temporada 2025 en vez de sustituir uno por otro: el retrospectivo de julio (`hilo_formula_campeon.md`, qué le pasó a cada equipo, 34 tuits) y el prescriptivo de agosto (`hilo_deberes_2026.md`, qué tiene que hacer cada equipo, 31 tuits). Los cuatro equipos de 12/12 van agrupados en un solo tuit en el prescriptivo.

**Motivo:** Son productos distintos, no versiones. El prescriptivo es más escaneable en un hilo largo porque el patrón "Deberes:" se reconoce a partir del tercer tuit, y encaja mejor en agosto, con la temporada a tres semanas. El retrospectivo sigue siendo útil si en algún momento se quiere contar la temporada cerrada.

**Alternativas descartadas:** Reescribir el de julio y perderlo → estaba auditado dato a dato, tirarlo era desperdiciar la verificación. Repartir el hilo en 17 días a 2 tuits/día hasta el arranque → Luis quiere hilo único; además, publicado en serie, cada tuit pierde el hilo narrativo y el tuit 1 tendría que anunciar una serie, no un hilo.
