# Backlog — ideas pendientes

Ideas verificadas contra los datos pero no construidas. Cada una dice qué la
sostiene y qué falta. Lo que se descarta se queda aquí con el motivo, para no
volver a proponerlo.

Lo que necesita suscripción a PFF vive aparte, en `docs/pff-wishlist.md`.

## El filtro, antes de proponer nada

**La pregunta no es "¿falta esto en el catálogo?" sino "¿lo tiene alguien
más?".** Que un hueco sea grande y fácil de llenar no lo convierte en contenido
que merezca la pena: si el lector lo encuentra a un clic en cualquier web
generalista, publicarlo no aporta nada. El valor de la cuenta está en lo que no
se encuentra en otro sitio.

Ese filtro ya se ha llevado por delante la clasificación (construida y
borrada), el parte de lesiones y el QBR de ESPN. Aplicarlo ANTES de escribir
código, no después.

Corolario útil: a veces el filtro no descarta la idea, la reencuadra. Los datos
de snaps en bruto son consulta; filtrados a posiciones de balón y contados como
"quién se ganó un puesto" son análisis. Antes de tirar algo, probar si hay un
corte que lo convierta en historia.

---

## Manning Bot

### `vegas_wp` como banco de pruebas — NO como feature
**Estado:** el punto 1 ya está construido (`--bench`, ago-2026); queda el 2.

Como **feature del modelo es redundante**: el `vegas_home_wp` de la primera
jugada correlaciona **0.996** con la probabilidad implícita del spread, y el
modelo ya usa `home_impl_prob`. Ya se descartó `spread_line` por el mismo
motivo (ver comentario en `FEATURE_COLS`). Meterlo sería la tercera copia del
mismo número y solo diluiría la importancia de las features.

Donde **sí** vale:

1. **Bot vs mercado — CONSTRUIDO (ago-2026), `python Manning_bot.py --bench`.**
   No hizo falta `vegas_wp`: el mercado ya está en el modelo como
   `home_impl_prob` (moneyline, cobertura del 100 % desde 2015), que es la
   misma información sin descargar nada.

   **Resultado, walk-forward 2022-2025 (953 partidos): el bot NO le gana al
   mercado.** 65,7 % de aciertos contra 68,2 %, y peor Brier (0,2154 vs
   0,2083) en las cuatro temporadas. Se separa del mercado en 96 partidos
   (10,1 %) y ahí acierta el **37,5 %** — o sea que cuando discrepa suele
   equivocarse. La discrepancia es ruido, no ventaja.

   **Consecuencia editorial:** no se puede publicar "el bot contra Las Vegas"
   como si fuera una ventaja, porque no la hay. Lo que sí es publicable es el
   dato honesto: un modelo con 30 features y once temporadas de histórico
   pierde contra una línea de apuestas. Y como diagnóstico interno vale doble:
   cualquier cambio futuro del modelo se mide contra este 37,5 %, que es la
   línea base que hay que superar para que el bot aporte algo propio.
2. **Curva de WP de `resumen_partido`.** Hoy usa `wp`, que ignora quién era
   favorito. `vegas_wp` lo incorpora. Verificado sobre 2025: difieren **más de
   10 puntos porcentuales en el 36,5 % de las jugadas** — no es cosmético.
   Para contar una remontada, la versión con línea cuenta mejor la historia.

---

### Mejorar la precision de Manning Bot — experimentos ago-2026
**Estado: RESUELTO. El resultado es la v6, que empata/supera al mercado:
68,9 % vs 68,2 % walk-forward 2022-2025, ganandole en 3 de 4 temporadas.**
Scripts de la bateria en `lab/manning_exp_bateria.py` (y
`manning_experimentos.py`, `manning_exp_qb_out.py`); las probs walk-forward
quedan en `lab/manning_probs_wf*.parquet` para futuras iteraciones.

Lo que funciono (las dos palancas suman +3,4 pts sobre la v5, del 65,5 %):

- **Regresion del margen de puntos** ademas del clasificador (+1,7 sola). El
  27-24 ensena mas que el gano/perdio; XGBRegressor+Ridge convertido a
  probabilidad con una normal, promediado con el clasificador de siempre.
- **Features de estabilidad** (+1,7 solas): success rate, EPA en situacion
  neutral (wp 5-95 %, fuera garbage time), EPA de pase en downs 1-2 y EPA de
  equipos especiales — fase que la v5 ignoraba por completo.
- Robusto a la semilla (68,2-69,0 % con seeds 7/42/777) y consistente por
  temporada. En discrepancias con el mercado (8,1 % de partidos) acierta el
  54,5 % — la v5 acertaba el 37,5 %. Muestra chica (77), no venderlo como
  ventaja sistematica; el Brier global sigue una pizca peor (0,2112 vs 0,2083).

Probado y descartado, con numeros:

- **Calibracion isotonica**: Brier empeora. El bot ya estaba bien calibrado;
  su problema era el objetivo y las features, no la escala.
- **Mezcla bot+mercado**: +0,5 pts ≈ ruido y peor Brier. Disfrazar al mercado.
- **Poda de features** (-8 menos importantes): 65,7 %, nada.
- **`d_qb_out` (titular Out/Doubtful del parte de lesiones)**: la señal existe
  (el equipo sin titular gana el 31,3 %) pero el parte solo cubre el 1,3 % de
  team-weeks (los IR no salen) y el bot YA sabe quien titula: `schedules` trae
  `home_qb_id`/`away_qb_id` y el QB rolling cruza por ahi. Global 65,5→65,4 %.
  Caveat: injuries_2020 y 2023 dieron 504 ese dia.

Diagnostico que guio todo (sigue valido): la v5 empataba con el mercado en
semanas 1-4 y 18 y perdia TODO el hueco en semanas 5-17; la v6 ahi da 69,6 %
vs 68,8 % del mercado.

Pendiente si algun dia se quiere mas: entrenar desde 2010 (moneyline al 100 %
hasta ahi, solo faltan los PBP 2010-2014), tuning de hiperparametros con CV
temporal, restricciones monotonicas en XGBoost.

---

## Columnas del PBP sin explotar

El PBP tiene 372 columnas y el proyecto usa 122. No requiere ninguna descarga
nueva.

| Columnas | Idea |
|---|---|
| `xyac_epa` · `xyac_mean_yardage` · `xyac_success` | YAC sobre esperado por receptor, sin depender de NGS |
| `cp` | Dificultad jugada a jugada (ya se usa `cpoe`, que es el agregado) |
| `fixed_drive_result` · `drive_time_of_possession` · `drive_play_count` · `drive_start/end_transition` | Análisis de drive real. `series_success` lo aproxima con cadenas de downs |
| `punt_attempt` · `kickoff_attempt` · `field_goal_attempt` · fair catches · `extra_point_result` | **Equipos especiales entero.** No falta la fuente: lleva en cada PBP descargado desde el principio |
| `two_point_attempt` · `replay_or_challenge_result` | Decisiones de entrenador: conversiones de 2 y acierto en los retos |
| `surface` · `no_score_prob` | Césped natural vs artificial; leverage real |

---

## Scripts nuevos

### Cruzar dline_presion_origen con oline_presion_origen en matchup_intel
Tenemos las dos mitades: por dónde presiona cada defensa y por dónde cede cada
línea ofensiva. Cruzarlas antes de un partido da el mismatch más directo que se
puede contar — "Green Bay mete el 77% de su presión por fuera y este ataque es
justo donde más cede". No necesita datos nuevos: los dos scripts ya calculan
sus tablas.

Limitación conocida: el cruce es por origen (interior/exterior/blitz), no por
lado ni por hueco — eso no existe en datos públicos (ver pff-wishlist item 9).

### Equipos especiales
Ni un script para kickers, punters ni retornadores. Fase entera sin cubrir y
sin competencia en español. Ver columnas del PBP arriba.

### PNG para MVPsSemana y MVPsSeason
Los dos calculan candidatos y solo los imprimen por consola. El cálculo ya está
hecho; falta dibujar. Abre además DPOY, novato del año y entrenador.

### Trayectoria de un jugador semana a semana
`season_arc` es de equipo. No hay equivalente de jugador.

---

## Fuentes cargadas pero sin visual

Los cargadores existen (ago-2026); faltan los gráficos.

- **`cargar_contratos`** — 51.952 contratos de OverTheCap con `apy`,
  `apy_cap_pct` y `guaranteed`, enlazados por `gsis_id` y con datos de draft.
  Rendimiento por dólar; cruza con los scripts de draft
- **`cargar_ngs`** — quedan RYOE (rushing) y YAC sobre esperado. La separación
  de receptor ya está en comparador_wrs
- **`cargar_snaps`** — quién ganó y quién perdió sitio a lo largo de la
  temporada. **Condición obligatoria: filtrar a QB/RB/WR/TE.** Sin filtrar, los
  diez que más suben son linieros que entraron por lesión del titular, o sea el
  parte de lesiones contado con otro número — y eso está descartado (ver abajo).

  Filtrado sí cuenta algo propio. Comprobado sobre 2025 (media de las semanas
  1-4 contra las 14-17): Dart 35→99 %, Bech 12→73 %, Tre Harris 22→65 %,
  TeSlaa 19→61 %; y en el otro sentido Dyami Brown 61→10 %, Calvin Austin
  74→31 %, Raymond 56→18 %. En Detroit el mismo movimiento se ve por los dos
  lados: TeSlaa sube 43 puntos mientras Raymond baja 38.

  Presentarlo como "quién se ganó un puesto", nunca como tabla de snaps.
  Aviso: es contenido de nicho y roza el terreno fantasy.

---

## Vigilancia

### `pbp_participation` en 2026
**Comprobar en la semana 2 de sep-2026** si existe `pbp_participation_2026`.
No tiene cron (es `workflow_dispatch`): en toda la temporada 2025 se ejecutó
una vez, en febrero, ya acabada. De ahí salen cobertura, personal y rutas de 13
scripts. `python estado_datos.py` lo reporta.

Si no aparece: cobertura y personal se quedan sin fuente gratuita, y PFF pasa
de lujo a plan B.

**Qué pasa exactamente si falta** (verificado ago-2026 ejecutando los scripts
con la carga de participación forzada a fallar):

- *Funcionan degradados, con aviso* — `oline_presion`, `oline_presion_origen`,
  `qb_presion`, `qb_overview`, `comparador_qbs`, `resumen_partido`. Solo usaban
  participación para la presión y tiran del proxy `qb_hit + sack`: pierden los
  hurries, así que la presión sale infravalorada, pero el PNG sale.
- *Paran con mensaje claro, sin gráfico* — `coberturas`,
  `ataque_por_personal_ofensivo`, `ataque_vs_personal_defensivo`,
  `defensa_vs_personal_ofensivo`, `red_zone_personal`, `matchup_intel`,
  `informe_equipo`. Necesitan personal o cobertura, que no tienen alternativa.
- *Los otros ~48 scripts no se enteran*: van contra PBP, stats_player,
  ftn_charting, PFR, NGS o snaps, todos con cron.

**Decisión (ago-2026): no construir la degradación parcial de `informe_equipo`
por adelantado.** Podría dibujarse sin las secciones de personal y cobertura
(los 3 KPIs, la franja de huecos, el bloque de presión y casi todos los
candidatos ocultos no dependen de participación), pero una tarjeta a medias es
peor tarjeta y el problema puede no llegar a darse. Se espera a ver qué pasa en
la semana 2 y, si no se actualiza, se busca vía entonces.

### `injuries`
El fichero se mantuvo toda la temporada 2025, pero el workflow localizado no
registra ejecuciones desde ago-2025 — probablemente cambió de repo. Confirmar
en vivo antes de montar contenido fijo encima.

---

## Descartado

- **Quién para la carrera en cada hueco (placadores) en el informe** (ago-2026)
  — el PBP trae el autor del placaje, así que se podría atribuir el hueco al
  edge más el apoyo de LB/safety. Descartado por densidad: la tarjeta de equipo
  ya lleva 3 KPIs, 4 secciones de ranking, la franja de huecos, identidad y el
  bloque de presión. Si algún día interesa, encaja en `run_gap_defensa`, no en
  el informe.

- **Clasificación / cuadro de playoffs** — construido y borrado el mismo día
  (ago-2026, commit `6e461c9`, revertido). Funcionaba: dos paneles AFC/NFC con
  los desempates oficiales de la NFL implementados y verificados a mano.

  **Motivo del descarte, de Luis:** *"eso no me parece relevante, quien quiera
  verlo entra a cualquier página y lo ve"*. Y tiene razón — la clasificación es
  el dato más disponible que existe. Republicarla no aporta nada que el lector
  no tenga a un clic, y el valor de la cuenta está en lo que NO se encuentra en
  otro sitio.

  **Lección para futuras propuestas:** que un hueco del catálogo sea grande y
  fácil de llenar no lo convierte en contenido que merezca la pena. El criterio
  no es "¿falta esto?" sino "¿lo tiene alguien más?". Antes de proponer una
  pieza nueva, comprobar que el dato no está ya en cualquier web generalista.

  Si algún día se recupera, el código está en el historial: `git show 6e461c9`.

- **Parte de lesiones** (ago-2026) — mismo criterio que la clasificación: lo
  publican los propios equipos cada miércoles y está en cualquier web. Un
  visual nuestro solo lo repetiría más tarde y más bonito. El cargador
  `cargar_lesiones` se conserva por si algún día hace falta como *contexto* de
  otro análisis ("este equipo cayó con 4 titulares fuera"), pero no como pieza
  propia.

- **Total QBR de ESPN** (ago-2026) — lo publica ESPN, literalmente. Además es
  una métrica cerrada de otra casa: no podemos explicar cómo se calcula ni
  defenderla si alguien la discute. `cargar_qbr` se conserva por si sirve para
  contrastar con nuestro EPA+CPOE, no para publicarla tal cual.

- **`vegas_wp` como feature de Manning Bot** — redundante con `home_impl_prob`
  (correlación 0.996). Ver arriba.
- **YPRR desde datos gratuitos** — haría falta rutas de TODOS los receptores;
  `pbp_participation` solo trae la del objetivo (una por jugada). Sigue siendo
  PFF (item 4 de la wishlist).
- **`ftn_charting` como sustituto de `pbp_participation`** — sus 29 columnas no
  incluyen cobertura, personal ni presión. Comprobado ago-2026.
