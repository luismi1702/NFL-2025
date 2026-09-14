# Decisiones técnicas — NFL 2025

---

## [2026-08-16] — El criterio para proponer contenido: «¿lo tiene alguien más?»

**Decisión:** Antes de construir una pieza nueva se comprueba que el dato no
esté ya a un clic en cualquier web generalista. La pregunta no es «¿falta esto
en el catálogo?» sino «¿lo tiene alguien más?».

**Motivo:** Se construyó `clasificacion.py` (cuadro de playoffs con los
desempates oficiales de la NFL implementados y verificados) y Luis lo mandó
borrar el mismo día: *"eso no me parece relevante, quien quiera verlo entra a
cualquier página y lo ve"*. Tenía razón — la clasificación es el dato más
disponible que existe. El error al proponerla fue valorarla por tamaño del
hueco y facilidad de construcción, no por si aportaba algo exclusivo.

**Alcance:** El filtro se llevó por delante también el parte de lesiones y el
QBR de ESPN. Corolario útil: a veces no descarta la idea sino que la reencuadra
— los datos de snaps en bruto son consulta, pero filtrados a posiciones de
balón y contados como «quién se ganó un puesto» son análisis.

**Alternativas descartadas:** Mantener `clasificacion.py` «porque ya está
hecho» → el coste hundido no cambia si aporta o no. Está en el historial
(`git show 6e461c9`) por si algún día se recupera.

---

## [2026-08-16] — Reparto porcentual vs tasa: el reparto engaña

**Decisión:** El origen de la presión y la carrera por hueco se miden en **tasa
por 100 dropbacks**, no como porcentaje del total del equipo.

**Motivo:** Luis lo detectó: *"si la media de la liga es 54% por el exterior y
el de SF 100%, ¿eso quiere decir que presionan mejor por el exterior? No,
quiere decir que está peor repartido, porque igual ese 100% son 2 presiones"*.
Comprobado sobre 2025 y las dos lecturas salen **opuestas**: por reparto SF es
«60% exterior contra 49% de la liga» (+11, parece bueno); por tasa es «10,1
contra 11,4» (−1,3, presiona MENOS por fuera). Por tasa, SF está por debajo de
la liga en los cuatro orígenes, coherente con ser 31º de 32 en volumen; el
reparto le pintaba dos en verde.

**Ventaja añadida:** las cuatro tasas SUMAN el KPI del bloque, así que «de esa
presión, el 60% viene por fuera» pasa a ser literalmente cierto.

**Alternativas descartadas:** Mantener el reparto añadiendo el volumen al lado
→ obliga al lector a hacer la corrección mental que el gráfico debería hacer.

---

## [2026-08-16] — Qué puede entrar en DONDE DOMINA / DONDE SUFRE

**Decisión:** Solo entran métricas donde **un ranking signifique mejor o peor**.
Quedan excluidas las de identidad (PROE, pass rate, uso de personal), que van a
la sección IDENTIDAD.

**Motivo:** Ser 1º en PROE no es bueno ni malo, es una forma de jugar. Si se
cuelan, la tarjeta acaba diciendo «domina en: pasar mucho», que no significa
nada. Todo lo que hay en el pool va contra EPA o contra una tasa de conversión,
que sí tienen dirección.

**Alcance:** El pool pasó de 4 secciones visibles a 19 candidatos, la mayoría
**ocultos** — no se dibujan en ninguna sección y solo aparecen si el equipo es
extremo. Antes el resumen era un resumen de lo que ya estaba a la vista y por
tanto no aportaba información nueva; peor, al mover la presión al bloque de la
derecha y meter los huecos como franja, la tarjeta mostraba dos cosas que su
propio resumen no veía.

---

## [2026-08-16] — Edge vs interior: depth chart y un corte de peso

**Decisión:** La clasificación de un rusher como interior o exterior sale de
`depth_chart_position` del roster, afinada con el peso: un DE de **280 libras o
más** cuenta como interior.

**Motivo:** Ni PFR ni nflverse sirven — Micah Parsons es `DL` para PFR y `LB`
para nflverse, y con esas etiquetas Green Bay salía como una defensa que
presiona por dentro cuando genera el 77% por fuera. El depth chart sí acierta
(le da OLB), pero no distingue el DE de un 4-3 (edge) del de un 3-4 (juega por
dentro): Zach Allen salía como exterior. El corte de 280 lb no es arbitrario —
sobre los 60 DE con ≥8 presiones en 2025, por debajo son edges sin discusión
(Will Anderson 243, Leonard Floyd 240) y por encima interiores puros (Leonard
Williams 302, Derrick Brown 318). Reclasifica 14 de 60.

**Alcance:** La misma regla en `dline_presion_origen` y `oline_presion_origen`,
que clasifican a los mismos defensores y tienen que coincidir o el «exterior»
de uno no es el del otro.

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

---

## [2026-08-24] — Copiar y pegar en vez de publicador automático

**Decisión:** No se construye el nivel 3 (publicador contra la API de X). En su
lugar, `cola_posts.py` deja los posts de la semana listos en una página HTML con
botón Copiar y el PNG a la vista; publicar sigue siendo copiar, pegar y arrastrar.

**Motivo:** La API de X es pay-per-use desde feb-2026 ($0.015/post, sin tier
gratis) y no aporta nada que el trabajo manual no dé: la verificación triple
antes de publicar es de Luis igualmente, o sea que el cuello de botella nunca
fue el clic de publicar, era buscar el texto y el PNG entre ~140 ficheros.

**Alternativas descartadas:** API de X → se paga por algo que no quita trabajo
real. TXT plano con separadores → hay que seleccionar a mano. Carpeta
`publicar/` con un fichero por post → empareja texto e imagen pero no ahorra el
copiado. Botón "Copiar imagen" → usa `navigator.clipboard.write`, que no existe
al abrir el HTML con doble clic (`file://`), justo el modo de uso previsto.

---

## [2026-08-24] — El formato de los borradores es un contrato, no prosa

**Decisión:** `borradores_prompt.md` obliga al redactor a un formato fijo
(sección `## `, línea `IMAGEN:`, cada alternativa dentro de un bloque cercado
`post`). `cola_posts.py` lo parsea con eso; si el redactor se desvía, la página
sale vacía y lo dice por consola en vez de fallar en silencio.

**Motivo:** El markdown libre de la primera versión no se podía parsear con
fiabilidad, y un parser que adivina produce tarjetas a medias — lo peor posible
en algo que existe para publicar sin releer.

**Alternativas descartadas:** Que el propio `claude -p` escriba el HTML → gasta
tokens en maquetar, es irrepetible entre corridas y no se puede verificar de
antemano. Parsear el markdown libre con heurísticas → falla en silencio.

**Coste:** El prompt y el parser quedan atados: tocar uno obliga a tocar el
otro. Anotado en `docs/calendario-posts.md`.

---

## [2026-08-24] — Ángulo negativo, sin mención  ·  DEROGADA el 14-sep-2026

> Sustituida por la entrada del 14-sep-2026: ahora se etiqueta siempre. Se
> conserva para saber qué se pensaba antes y por qué cambió.

**Decisión:** Si un post cuenta la caída, la mala racha o el fracaso de un
equipo, van los hashtags pero ninguna mención a su afición.

**Motivo:** Las menciones existen como palanca de alcance — se etiqueta para que
la afición comparta. Nadie comparte el post que certifica su hundimiento, así
que la mención no solo no suma: se lee como pulla.

**Alcance:** Regla general en `CLAUDE.md`, `borradores_prompt.md` (regla 9),
`docs/cuentas-fans.md` y `docs/post-ejemplos.md`. **Excepción decidida por Luis:**
el hilo "Deberes 2026" se etiqueta entero, negativos incluidos, porque el hilo
es prescriptivo de cabo a rabo; queda escrito en la cabecera del propio hilo.

---

## [2026-09-11] — Al arrancar la temporada, la norma es el año anterior

**Decisión:** Las "Claves del partido" de `resumen_partido.py` comparan el
partido con el resto de la temporada, pero si el equipo lleva menos de 3
partidos (semanas 1-3) la norma pasa a ser la temporada regular anterior
completa. El PNG lo rotula en el subtítulo ("vs su temporada 2025") y en el pie
("plantillas cambian"), y cada línea mantiene su `n`.

**Motivo:** En la semana 1 el "resto de la temporada" está vacío y los dos PNGs
salían enteros con "Sin desviaciones con muestra suficiente" — el peor
resultado posible en la pieza que el calendario manda publicar el martes. La
alternativa de no comparar deja el visual sin la mitad de su contenido justo en
la semana con más audiencia del año.

**Coste aceptado:** entre años cambian plantillas y entrenadores, así que la
desviación mezcla "hicieron algo distinto" con "ya no son el mismo equipo".
Por eso el año va rotulado en la imagen y no escondido en el código: el lector
ve contra qué se compara. En el partido SEA-NE la norma de Seattle es de un
equipo que aún tenía a Darnold de titular.

**Alternativas descartadas:** Comparar con la media de la liga → mide algo
distinto (bueno/malo, no "distinto a lo suyo"), que es justo lo que el resto de
la pieza ya cuenta. Dejar el hueco vacío hasta la semana 4 → tres semanas sin
segundo PNG. Bajar el mínimo de jugadas de la norma → con un partido la
"norma" es ruido presentado como costumbre.

**Alcance:** En la misma sesión, `resumen_partido.py` acepta siglas
alternativas (`LAR`→`LA`, `JAC`, `WSH`, `LVR`…) porque desde la galería web
`LAR` moría con "No se encontro el partido"; el error lista ahora los partidos
disponibles de la semana.

---

## [2026-09-14] — Los resúmenes se nombran por orden de kickoff

**Decisión:** Los PNGs de `resumen_partido.py` pasan a llamarse
`NN_resumen_VIS_vs_LOC_{año}_wNN.png`, donde NN es el orden de kickoff de la
jornada según el calendario (01 = partido inaugural del miércoles, 15 = Sunday
Night) y los equipos van siempre visitante_vs_local. Lo resuelve
`orden_partido(season, week, *equipos)` en `pbp_loader`; si el calendario no se
puede consultar, se cae al nombre de siempre en vez de fallar.

**Motivo:** Con 16 partidos por jornada son 32 PNGs sueltos en la carpeta de la
semana, y el orden alfabético los baraja: el Monday Night podía salir el
primero. Luis lo pidió explícitamente — que la carpeta se lea como se jugó la
jornada, con los dos PNGs de cada partido juntos. El orden fijo
visitante/local arregla de paso que el nombre dependiera de en qué orden se
tecleaban las siglas: convivían `resumen_LA_vs_SF` y `resumen_SF_vs_LA` del
mismo partido, o sea dos ficheros para un solo análisis.

**Alternativas descartadas:** Una subcarpeta por partido → agrupa mejor, pero
obliga a entrar y salir de 16 carpetas para arrastrar imágenes a X, que es el
trabajo manual que el nivel 2.5 existe para recortar. Prefijo con día y hora
(`dom1300_…`) → más informativo pero más largo, y el número ya da el orden.

**Alcance:** Solo `resumen_partido.py`, que es el único script con un PNG por
partido. `salida()` no se toca: el prefijo lo construye quien llama.

---

## [2026-09-14] — Se etiqueta siempre (deroga "ángulo negativo, sin mención")

**Decisión:** Todo post sobre un equipo concreto lleva la mención a su afición,
gane o pierda, también cuando cuenta la derrota o la mala racha. Queda una
sola excepción: los equipos marcados *(pendiente)* en `docs/cuentas-fans.md`
(hoy ATL e IND), que no tienen cuenta conocida — y ahí sigue en pie que no se
inventan ni se deducen handles.

**Motivo:** Petición directa de Luis: *"Quita esa norma, se etiqueta siempre"*.
Es su cuenta y su criterio editorial. La regla de agosto nació de una hipótesis
razonable —nadie comparte el post que certifica su hundimiento— pero nunca se
midió: las visualizaciones que la justificaban comparaban tuits CON y SIN
mención, no tuits positivos contra negativos.

**Lo que se pierde:** el argumento de agosto sigue siendo plausible, así que
conviene mirar el rendimiento de los posts negativos etiquetados de la semana 1
antes de darlo por bueno del todo. Si una cuenta se queja, la decisión se
revisa; con 32 aficiones el coste de equivocarse es reputacional, no de
alcance.

**Alcance:** `CLAUDE.md`, `borradores_prompt.md` (regla 9), `docs/cuentas-fans.md`,
`docs/post-ejemplos.md` y la entrada del 24-ago-2026, marcada como derogada
pero conservada: el registro existe para saber qué se pensaba antes.

---

## [2026-09-14] — Qué se copia de un boletín de pago y qué no

**Decisión:** De la newsletter de SumerSports se toman **ideas y ángulos**, nunca
cifras. Cualquier dato que se publique se recalcula con nuestro PBP; lo que no
se pueda recalcular, o se cita como suyo o no se usa. No se paga SumerPass
(100 $/año): se mantiene el plan de PFF+ anual.

**Motivo:** Dos hallazgos al contrastar sus 14 fichas contra nuestros datos.
Primero, **usan su propio modelo de EPA**: en los 18 ataques comparados el
nuestro sale siempre más alto, entre 0,009 y 0,080. No es ruido, es otro
modelo, así que mezclar sus cifras con las nuestras produciría un gráfico
incoherente consigo mismo. Segundo, **no documentan exportación** de datos en
ninguna parte de su web, y el flujo del proyecto es CSV → `pff_data/` →
scripts. Sin exportación, 100 $/año compran lectura, no datos. PFF+ cuesta lo
mismo (99,99 $, no los 79,99 $ que decía la wishlist) pero sí exporta CSV.

**Lo que sí se replicó:** de las 28 casillas de sus fichas reproducimos 17 con
nflverse. Ocho son imposibles sin charting (presión, YPRR, play action,
personal, yardas antes del contacto, cobertura por defensor) y tres tienen
definición divergente que no conviene imitar a ciegas: PASS OE (su modelo de
pase esperado no es el `xpass` de nflverse), NEG% y MOF% (el nuestro sale justo
el doble: ellos miden por posición del receptor, no por `pass_location`).

**Alternativas descartadas:** Pagar el mensual de 20 $ para "ir mirando" → son
240 $ por temporada, más caro que el anual y sin resolver la exportación.
Copiar sus números citando la fuente → rompe la verificación en tres frentes
del `CLAUDE.md`, que exige poder reproducir cada cifra.

**Consecuencia:** nacen `under_center.py` y `ficha_tactica.py`, que cubren con
datos propios lo mejor de su producto. La ficha usa **percentiles contra los
partidos de equipo de la temporada anterior** (544 en 2025) en vez de contra la
temporada en curso, que en la semana 1 no existe; y las métricas de identidad
(bajo centro, scrambles) van sin percentil ni color, por la regla de jul-2026:
solo se juzga lo que tiene dirección.

