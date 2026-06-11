# Decisiones técnicas — NFL 2025

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

**Decisión:** `contenders_tracker.py` descarga sus propios datos frescos (schedules + PBP con cache incremental por semana) y calcula las 12 métricas directamente del PBP. Umbrales recalibrados al peor campeón con estas definiciones (def_epa ≤-0.015, pts cedidos pace ≤378, 3ª bajada ≤0.415, sacks pace ≤45.5, yds/jugada ≤5.85). Re-verificado 11/11 campeones 2015-2025.

**Motivo:** Los caches congelados (game_logs, player_stats, newmetrics) no se actualizaban nunca → el tracker habría dicho "sin datos" en 2026. Además las filas QB 2025 de player_stats tenían `sacks=0.0`, lo que invalidaba el criterio de sacks. La latencia ya no es problema: el PBP se cachea y solo se re-descarga con jornadas nuevas.

**Alternativas descartadas:** Script semanal separado de actualización de caches (`actualizar_semana.py`) → el usuario prefirió apilar todo en el script original. Mantener umbrales antiguos → 6/11 campeones fallaban por centésimas con las definiciones corregidas.

---

## [2026-06-12] — pbp_loader.py: única excepción a "scripts independientes"

**Decisión:** Módulo común `pbp_loader.py` (`cargar_pbp/cargar_stats/cargar_participation`) que todos los scripts importan. Cache local del parquet oficial de nflverse, filtro REG por defecto, auto-detección de temporada.

**Motivo:** 45 scripts duplicaban el mismo bloque de carga con 3 defectos sistémicos (playoffs mezclados en stats de temporada, re-descarga de 90MB por ejecución, año hardcodeado). Arreglarlo en un solo sitio frente a mantener 45 copias.

**Alternativas descartadas:** Copiar el bloque corregido en cada script (mantenía la independencia total pero multiplicaba el mantenimiento ×45). Los scripts de partido/semana concreta usan `solo_reg=False` para no perder acceso a playoffs.
