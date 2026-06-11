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
