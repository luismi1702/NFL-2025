# Auditoría visual de PNGs — 2026-07-17/18

## PASADA EDITORIAL (2026-07-18) — APLICADA
Segunda revisión con dos preguntas por visual: ¿qué sobra? y ¿qué falta que el lector esperaría?
Origen: el usuario cazó dos casos que la primera pasada (centrada en defectos de render) no marcó
— paneles duplicados en tendencias_playcalling y rankings ausentes en matchup_intel.

**Sobraba (redundancia de tinta):**
- qb_presion: el colorbar codificaba el eje Y → ahora color = % de dropbacks bajo presión (info nueva).
- power_rankings y valor_turnovers: colorbars eliminados (color = valor de la propia barra).

**Faltaba:**
- resumen_partido: **marcador final en el título** (SF 17-13 SEA).
- proe: "real X% · esp Y%" junto a cada barra (el PROE solo es la diferencia).
- play_action: "PA +0.42 · sin +0.00 · uso 22%" junto a cada barra.
- power_rankings: récord W-L junto al score (desde schedules cache, con fallback silencioso).
- Previas: récord W-L junto a cada logo.
- QBsTotalEPA: unificado con ranking_* — líneas en la media de la muestra + tamaño de logo por volumen.
- Radares comparadores (×9): el equipo del jugador en la leyenda ("M.Garrett (CLE)").
- run_gap/run_gap_defensa: etiquetas LE/RE en las flechas exteriores (los círculos OL ya tenían fallback de posición).
- red_zone_personal grid: partido en 2 PNGs (ataque/defensa) con texto de celda legible; galería actualizada.

Lección incorporada: la revisión de visuales necesita dos lentes — defectos de render Y editorial
(cada elemento debe ganarse el sitio; cada ausencia que el lector espera es un bug).

Análisis imagen a imagen de los PNGs generados por cada script.

**ESTADO: P1 y P2 APLICADOS y verificados (2026-07-17).** Cambios clave:
- Logos normalizados por píxeles (`zoom = base * 500 / max(h,w)`) en los 39 scripts; HARD_PENALTY eliminado. NYJ es un wordmark oficialmente (el PNG "escudo" no existe): ahora ocupa el mismo lienzo que los demás.
- `sys.stdout.reconfigure(utf-8)` en los 5 comparadores defensivos; `socket.setdefaulttimeout(30)` en pbp_loader, Manning_bot y contenders_tracker.
- red_zone_personal: normalización por percentiles 5-95 + columnas de personal vacías eliminadas.
- Esquina inferior derecha: leyenda de volumen de ranking_* sustituida por nota al pie; marca de agua a nivel de figura en QBsTotalEPA y ranking_*.
- Anti-solape de etiquetas (`separar_etiquetas`, greedy vertical) en qb_overview, qb_presion, QBsTotalEPA y ranking_wrs/rbs/tes; en qb_presion las etiquetas van ahora bajo el punto y se quitó la diagonal parcial.
- power_rankings: texto OF/DF por luminancia; RankingEPAadjustado y DatoSemana: valores negativos fuera de la barra; yticks fantasma eliminados en 6 barh; coberturas/contenders/resumen: textos recolocados; series_success: título "series" + espaciado compacto; resumen: el QB ya no puede salir como "RB".

**P3 APLICADO (2026-07-18)** con decisiones del usuario:
- draft_value_cliff: rediseñado como small multiples 2×4 (una mini-gráfica por posición, resto en gris de contexto, ↓pp = mayor caída). El modo "posición" individual se mantiene.
- Radares defensivos: la 6ª métrica pasa de EPA on/off a **producción por partido** — edges/DTs "Disrupción/PJ" (sacks+hits+TFL)/PJ, LBs "Producción/PJ" (tackles+TFL+sacks)/PJ, CBs "PD+INT / PJ", Safeties "Impacto/PJ" (tackles+PD)/PJ. Se eliminó la carga de pbp_participation (los scripts van mucho más rápido; cbs/lbs/safeties tampoco cargan ya el PBP).
- draft_success: colormap escalado a 0-50% (leyenda actualizada); draft_ranking: xlim ajustado al rango real.
- tendencias_playcalling: MIN_PLAYS 10→25 (celdas con muestra ínfima salen n/d). Además (2026-07-18, a petición del usuario): de 4 paneles a 2 — los paneles "Delta vs liga" duplicaban visualmente los de arriba; ahora cada celda muestra valor + Δ liga + n y el color representa el Δ.
- Texto de celda por luminancia real del fondo en los 6 heatmaps (vs_personal ×3, run_gap ×2, series_success).
- clutch grid: etiquetas de cuadrante a las esquinas; game_script grid: texto de la diagonal eliminado.
- short_name descarta sufijos (Jr./Sr./II...) en ranking_*, QBsTotalEPA y qb_overview ("Deebo Samuel Sr." → "D. Samuel").
- season_arc: fondo más tenue y conector punteado siempre visible hasta el logo.

## Problemas TRANSVERSALES (afectan a muchos scripts)

1. **Logo NYJ aplastado** — `HARD_PENALTY = {"NYJ": 4.5}` en ~25 scripts encoge el wordmark JETS hasta hacerlo ilegible (sale como texto verde diminuto). El logo actual de `logos/NYJ.png` es un wordmark 4096px muy ancho; la penalización 4.5 era para otro archivo. Fix: normalizar TODOS los logos por píxeles reales (`zoom = TARGET_PX / max(h, w)`) como ya hace `coberturas.py`, y eliminar HARD_PENALTY. Un solo cambio de patrón en todos los scripts iguala también LAC/BAL/PHI/DAL, que salen más pequeños que el resto por el heurístico de aspect ratio.
2. **Encoding Windows (cp1252)** — `comparador_cbs/lbs/dts/safeties` CRASHEAN en consola al imprimir "≥" (`≥`). Mismo bug que tenía discriminacion_total. Fix: `sys.stdout.reconfigure(encoding="utf-8")` al inicio (o PYTHONIOENCODING). Revisar también otros scripts que impriman ≥/★.
3. **Guiones de yticks fantasma** — en muchos barh (`power_rankings`, `valor_turnovers`, `coberturas`, `oline_presion`) los yticks vacíos dejan un "-" visible a la izquierda. Fix: `ax.set_yticks([])` en vez de labels vacíos.
4. **Marca de agua vs. esquina inferior derecha ocupada** — cuando hay leyenda/etiqueta de cuadrante en esa esquina, colisionan (ver ranking_wrs, QBsTotalEPA).

## Por script

### qb_overview.py
- Clúster central ilegible: T.Shough/Herbert/L.Jackson/C.Wentz/A.Rodgers/M.Mariota labels superpuestos; M.Jones pisa J.Allen; D.Prescott/J.Goff/D.Jones montados. Fix: separación vertical de etiquetas (algoritmo greedy como season_arc) o etiquetar solo fuera del clúster.
- NYJ/WAS diminutos (transversal #1).

### QBsTotalEPA.py
- **Marca de agua pisa la etiqueta de cuadrante** "Bueno en RZ / Malo 3ro" (inferior dcha).
- Logos demasiado pequeños en general (base_zoom 0.030 en figura 13×9) y desiguales.
- C.Ward/J.McCarthy y A.Rodgers/S.Darnold solapados; D.Maye/B.Nix/J.Allen apiñados.

### ranking_wrs.py (y ranking_rbs/tes por plantilla)
- **Esquina inferior derecha: 4 elementos apilados** — leyenda de volumen + etiqueta de cuadrante + marca de agua + el punto "M. Brown". Fix: leyenda arriba-izda o fuera del axes; cuadrante inferior-dcho sin texto si hay leyenda.
- Solapes de nombres en el centro (C.Kupp/Sutton, N.Collins/O.Zaccheaus, J.Meyers/X.Legette, M.Washington/T.Franklin, A.Brown ×2).

### oline_presion.py
- Bien tras FTN. Solo NYJ (transversal) y el label "Liga avg" toca el value label de la primera fila.

### comparador_qbs.py (radar, todas las posiciones)
- Correcto. Etiquetas laterales (CPOE / EPA pocket limpio) rozan las spoke lines. Menor.

### series_success.py
- Título aún dice "Eficiencia de drive" → debe decir "de series".
- Mucho espacio vacío arriba y abajo (fig_h sobredimensionado).
- El separador vertical tras Conv% sobresale por debajo de la tabla.
- NYJ transversal.

### tendencias_playcalling.py
- Pass% panel OK tras fix Normalize(0,1).
- Celdas con n=10-16 (1er down corta/media) destacan igual que n=456. Fix: atenuar alpha o marcar "muestra baja" bajo n<25.

### run_gap.py (fan chart)
- "n=119" del hueco C queda pisado por el círculo del center; n= de los huecos interiores pegados a los círculos OL.
- Flechas LE/RE sin etiqueta — añadir "LE"/"RE" pequeñas.
- Dorsales OL no aparecen (depth_charts 2025 falla silenciosamente) — círculos vacíos.

### coberturas.py
- **"Fuente: FTN Data..." está superpuesta con la leyenda** (ilegible). Mover la fuente abajo.
- COVER 0 y COVER 1 casi mismo rojo — difícil distinguir segmentos contiguos.
- NYJ transversal. Marca de agua pegada a la última barra.

### target_share.py
- RACR de "Otros" = 11.87 (air yards netos ≈ 0) — mostrar N/D si air_yards_sum < umbral (p.ej. 25).
- Guiones yticks fantasma en panel derecho.

### power_rankings.py
- **Texto "OF:/DF:" dentro de las barras casi ilegible** (gris claro sobre amarillo). Fix: color de texto según luminancia de la barra (negro sobre claro).
- Guiones yticks; NYJ transversal.

### cuarto_down.py
- Correcto. Espacio vacío arriba/abajo mejorable; "Liga" label roza la línea.

### valor_turnovers.py
- Correcto. Logos en columna fija lejos de las barras negativas (aceptable); guiones yticks; NYJ.

### matchup_intel.py
- Muy bien. Sin cambios.

### informe_equipo.py
- El marcador "|" de media de liga a veces pisa el texto centrado en la barra (+0.143 Hombre, n=906 Pocket limpio). Fix: dibujar el marcador con alpha o desplazar el texto si coincide.

### clutch_performance.py (scatter 32)
- Etiquetas de cuadrante demasiado cerca del centro (22% del rango): "VULNERABLES"/"BUENOS ATQ." caen en medio del enjambre. Moverlas a las esquinas.
- Clúster central de logos + flechas ruidoso (inherente); NYJ transversal.

### season_arc.py
- 6 líneas destacadas = spaghetti; bajar a 3-4 por defecto o auto top3+bottom3 con más alpha en el resto.
- Conector fino hasta el logo confunde (parece otra serie); logos de la derecha lejos de su línea cuando hay anti-solape.
- NYJ transversal.

### mapa_pases_por_zona.py
- Correcto tras fix pass_touchdown. Franjas de textura crean dos tonos por celda (leve confusión, estético).

### ataque_por_personal_ofensivo (renombrado jul-2026).py (y variantes por plantilla)
- "n=" blanco poco legible sobre celdas saturadas (BUF +0.394, SEA -0.301, CIN +0.365). Texto negro también en esas.
- Espacio vacío arriba; NYJ transversal.

### draft_success.py
- Colorbar 0-100% pero máximo real 44% → todo cae en el tercio rojo, el verde no se usa. Escalar vmax≈50%.
- "Fuente" roza los labels del colorbar.

### qb_presion.py
- **Solapes severos de nombres**: labels centrados SOBRE el punto (va=center) → T.Lawrence/Herbert/Darnold, L.Jackson/J.Hurts, C.Ward/P.Rivers, K.Murray/M.Penix/J.Fields montados. Fix: offset vertical del label + anti-solape.
- Diagonal y=x flota en la esquina sup-izda sin guiar (rangos muy asimétricos) — quitarla o dibujarla completa.
- "Problemas en todo" pisa el punto M.Brosmer.

### DatoSemana.py
- Pie fijo "Solo pases y carreras" contradice cuando la métrica ganadora es de equipos especiales → hacerlo dinámico.
- Signo "-" de valores negativos pequeños queda tapado por la barra/eje (PHI -0.014 se lee "0.014").
- NYJ transversal.

### RankingEPAadjustado.py
- **Valores negativos: el texto queda DENTRO de la barra** y en los cercanos a 0 el "-" se corta. Fix: ha="right" y x a la izquierda de la barra para v<0.
- NYJ transversal.

### comparador_edges/cbs/lbs/dts/safeties (radares defensivos)
- CRASH cp1252 por "≥" en consola (transversal #2) — sin `sys.stdout.reconfigure` no generan nada fuera del dashboard.
- **EPA on/off casi siempre N/D o ~0.00 con MIN_SNAPS=100** (titulares no tienen 100 snaps "off") → la sexta métrica del radar es inútil. Propuesta: sustituir por métrica per-game (sacks/PJ o presiones FTN del jugador si hay datos) o volver a un mínimo intermedio (50) asumiendo ruido.
- Etiquetas laterales del radar pisan las spoke lines (menor, igual que QBs).

### run_gap.py (heatmap 32)
- Espacio vacío arriba; "n=" blanco poco legible sobre celdas medias; NYJ transversal.

### play_action.py
- Correcto. Espacio muerto a la derecha (xlim holgado); guiones yticks; NYJ transversal.

### red_zone_personal.py (grid 32)
- **Outlier con n mínimo revienta la escala de color**: SEA 21-personal -1.95 (n=10) sale verde brillante y aplana el resto del panel defensivo en naranja. Fix: normalizar con percentiles (vmin/vmax robustos) o subir el mínimo de snaps del grid.
- Columna "10 personal" completamente vacía en ambos paneles → eliminarla si no hay datos.
- Textos de celda diminutos en relación al tamaño del PNG (ilegible en móvil); valorar partir en 2 PNGs (ataque/defensa).

### resumen_partido.py
- **"Fuente:" pisa la jugada #3 del Top 3 WPA** (esquina inferior izquierda, ilegible).
- **El "RB" líder puede ser el QB** (scrambles cuentan como carrera): SEA muestra "RB: S.Darnold". Excluir al QB del apartado RB.
- Espacio vacío grande entre las cards y el Top 3.

### Previas.py (modo partido)
- Impecable. Solo falta línea de fuente al pie.
- CLI: pide 4 inputs (p/j, A, B, Semana) — con 3 por stdin lanza EOFError; el dashboard ya manda 4.

### contenders_tracker.py
- **Leyenda inferior rota**: los chips de color se recortan/quedan pisados por "Fuente:...". Solo se leen "3+ metricas"/"sin datos" sueltos. Recolocar leyenda dentro del área.
- Cabeceras "Y/intento pase"/"Y/jugada cedidas" poco legibles — dejar "Yds/int" y "Yds/jug".

### clutch_performance.py / game_script.py (modo equipo)
- Correctos. Sin cambios necesarios.

### draft_ranking_equipos.py
- **Eje X hasta 80% con máximo real 23%** — dos tercios del lienzo vacío. Ajustar xlim al dato.
- Label "Media: 15.3%" pisa el borde superior; "n=" atravesado por la línea de media en algunas filas.

### draft_value_cliff.py
- Spaghetti de 8 líneas cruzadas; las anotaciones "↓%" flotan sin asociación clara. El título pregunta algo que el visual no responde. Propuesta: small multiples por posición (2×4) o resaltar 2-3 posiciones y atenuar el resto. Es rediseño, no fix.

### Revisados por plantilla equivalente (sin issues nuevos): ranking_rbs/tes (=ranking_wrs), comparador wrs/rbs/tes/cbs/lbs/dts/safeties (=comparador_qbs/edges), ataque_vs_personal_defensivo y defensa_vs_personal_ofensivo (=ataque_por_personal_ofensivo (renombrado jul-2026)), run_gap_defensa (=run_gap), season_arc defensa (=ataque), proe (=oline_presion), draft_grid/eval_2026/success_r1 (=draft_success), draft_r1_radar.

---

## PLAN DE FIXES PRIORIZADO

**P1 — Rotos o engañosos**
1. Encoding cp1252: `sys.stdout.reconfigure(utf-8)` en comparador_cbs/lbs/dts/safeties/edges (crashean en consola).
2. Timeout de red en pbp_loader (`socket.setdefaulttimeout(30)` + timeout en read_csv de schedules): un cuelgue de red congela cualquier script para siempre (pasó en la auditoría).
3. red_zone_personal: normalización robusta del color (percentiles) + quitar columna "10 personal" vacía.
4. coberturas: fuente fuera de la leyenda. contenders: leyenda dentro del lienzo. resumen_partido: fuente fuera del Top-3 + RB ≠ QB.
5. series_success: título "series", compactar espacios, cortar el separador.

**P2 — Sistémicos de calidad visual**
6. Logos: normalización por píxeles (TARGET_PX/max(h,w)) en todos los scripts, fuera HARD_PENALTY → arregla NYJ + iguala LAC/BAL/PHI/DAL en ~25 scripts.
7. Esquina inferior derecha: en ranking_* mover leyenda de volumen arriba-izda; en QBsTotalEPA subir la marca de agua o quitar la etiqueta del cuadrante ocupado.
8. Anti-solape de etiquetas en scatters (qb_overview, qb_presion, QBsTotalEPA, ranking_*): offset bajo el punto + separación vertical greedy.
9. power_rankings: texto OF/DF en color según luminancia. RankingEPAadjustado/DatoSemana: valores negativos fuera de la barra.
10. Guiones fantasma de yticks (set_yticks([])) en los barh.

**P3 — Mejoras de diseño**
11. draft_success/draft_ranking: escalar colormap/xlim al rango real.
12. informe_equipo: marcador "|" de liga sin pisar el texto.
13. clutch grid: etiquetas de cuadrante a las esquinas. game_script grid: diagonal sin texto sobre logos.
14. tendencias/ataque_vs_personal: atenuar celdas con n bajo; "n=" siempre en color legible.
15. season_arc: 3+3 destacados máx con conectores más visibles. draft_value_cliff: rediseño small-multiples.
16. Radar defensivo: sustituir EPA on/off (N/D con MIN_SNAPS=100) por métrica per-game.
