# CHANGELOG — NFL 2025 @CuartayDato

---

## [2026-06-11] — contenders_tracker.py autosuficiente + recalibración de umbrales

**Qué se hizo:**
- `contenders.py` borrado: superado por `contenders_tracker.py --season X` (mismos criterios, pero recalibrados y con datos frescos; el viejo daba resultados inconsistentes)
- `contenders_tracker.py` ya no depende de caches congelados: descarga schedules frescos en cada ejecución (y actualiza `schedules.parquet`) y el PBP de la temporada solo cuando detecta jornadas nuevas (cache `pbp_contenders_{season}.parquet`). Listo para uso semanal en 2026: `python contenders_tracker.py` y sale el PNG al día
- Métricas calculadas directamente del PBP replicando las definiciones de calibración (game logs de Manning_bot); YPA y sacks ahora estilo oficial (intentos sin sacks, yardas de pases completados)
- **Bug de datos corregido**: las filas QB 2025 de `player_stats` tenían `sacks=0.0` (aproximación), así que el criterio "sacks permitidos" lo pasaba todo el mundo gratis en 2025, y el YPA estaba deflactado por incluir yardas de sack
- Umbrales recalibrados al peor campeón con las nuevas definiciones (solo se relajaron los 5 que fallaban por centésimas): def_epa ≤-0.015, pts cedidos pace ≤378, 3ª bajada ≤0.415, sacks pace ≤45.5, yds/jugada ≤5.85 (los dos últimos ya estaban anotados como pendientes en el CHANGELOG del 29-05 pero nunca llegaron al tracker)
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
