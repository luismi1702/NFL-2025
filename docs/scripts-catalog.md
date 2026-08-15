# NFL 2025 — Catálogo de scripts

## Machine Learning
| Script | Descripción |
|--------|-------------|
| Manning_bot.py | Predictor resultados. Ensemble XGBoost+RF+LR (pesos 3:1:2). Walk-forward CV 2015-2024. ~67-69%. Modelo en manning_bot_model.pkl |

## Utilidades / Cache
| Script | Descripción |
|--------|-------------|
| generar_caches.py | Regenera los caches por año (fullmetrics, epa_type, situational, explosive, rz_def, newmetrics + QBs en player_stats). Uso: `python generar_caches.py 2026 [--force]` |
| server.py | Backend mínimo para ejecutar scripts desde el navegador (http://localhost:8765) |

## QBs
| Script | Descripción |
|--------|-------------|
| comparador_qbs.py | Radar chart dos QBs: 6 métricas EPA + índice EPA+CPOE |
| QBsTotalEPA.py | Scatter EPA/play Red Zone vs 3er down |
| qb_presion.py | EPA bajo presión vs pocket limpio |
| qb_overview.py | Scatter todos los QBs: EPA/play vs CPOE, tamaño = intentos, color = EPA bajo presión |
| mapa_pases_por_zona.py | Mapa de pases por zona del campo |

## Ataque
| Script | Descripción |
|--------|-------------|
| comparador_wrs.py | Radar comparativo WRs |
| comparador_rbs.py | Radar comparativo RBs |
| comparador_tes.py | Radar comparativo TEs |
| ranking_wrs.py | Scatter EPA/objetivo Red Zone vs 3er down — WRs |
| ranking_rbs.py | Scatter EPA/objetivo Red Zone vs 3er down — RBs |
| ranking_tes.py | Scatter EPA/objetivo Red Zone vs 3er down — TEs |
| target_share.py | Target share, air yards share, WOPR, RACR por receptor |
| play_action.py | EPA/pase con play-action vs sin play-action por equipo (is_play_action de FTN charting, solo 2022+) |
| series_success.py | Heatmap 32 equipos: % de series por resultado (serie = cadena de downs; TD/FG/1st/Punt/Fallo) |
| tendencias_playcalling.py | Pass% y EPA por down × distancia, comparado vs media NFL |

## Carrera
| Script | Descripción |
|--------|-------------|
| run_gap.py | Heatmap EPA/acarreo por dirección de hueco. Solo acarreos diseñados |
| run_gap_defensa.py | Igual desde perspectiva defensiva. Verde = buena defensa |

## Defensa
| Script | Descripción |
|--------|-------------|
| comparador_edges.py | Radar Edge Rushers: sacks, QB hits, TFL, fumbles, EPA |
| comparador_cbs.py | Radar comparativo CBs |
| comparador_lbs.py | Radar comparativo LBs |
| comparador_dts.py | Radar comparativo DTs |
| comparador_safeties.py | Radar comparativo Safeties |
| oline_presion.py | Tasa de presión permitida por línea ofensiva |
| oline_presion_origen.py | Por dónde cede presión cada OL. Enter = heatmap 32 (presión FTN + origen: interior DT/NT, exterior DE/OLB, blitz LB/DB, vía atribución de sacks+QB hits). Sigla = diagrama de campo con flechas de origen y los titulares de cada puesto. Nota: la presión cedida POR liniero concreto (LT vs RT) no existe en datos públicos — es charting de pago (PFF Premium Stats / SIS / FTN StatsHub) |
| cuarto_down.py | Conversiones en 4º down |
| power_rankings.py | Composite: 40% EPA ofensivo + 40% defensivo + 20% tendencia ofensiva reciente (sin ajuste por rival) |
| clutch_performance.py | Rendimiento en situaciones cerradas (4Q + OT, ≤7 pts). Un equipo o scatter 32 |

## Cobertura y personal
| Script | Descripción |
|--------|-------------|
| coberturas.py | Distribución coberturas defensivas (Cover 0-6). Datos FTN 2024 |
| ataque_por_personal_ofensivo.py | Uso y eficiencia de paquetes ofensivos propios (renombrado jul-2026: antes "ataque_vs_..." — el "vs" era engañoso, mide el ataque CON su propio personal) |
| ataque_vs_personal_defensivo.py | Eficiencia ofensiva según el personal defensivo del rival |
| defensa_vs_personal_ofensivo.py | Heatmap EPA permitido según el personal ofensivo del rival |
| red_zone_personal.py | EPA en red zone según personal ofensivo. Un equipo o heatmap 32 |

## Análisis de equipo / matchup
| Script | Descripción |
|--------|-------------|
| informe_equipo.py | Team card de un equipo: 2 PNGs (ataque y defensa). Rediseñado jul-2026: 4 KPIs con rank, cada faceta (personal, coberturas, man/zona, presión) como punto en pista de ranking 1→32, fortalezas/debilidades autogeneradas (top/bottom 25% del nº real de equipos con muestra) e identidad (uso vs media NFL) |
| matchup_intel.py | Intel táctica A vs B — mismatches automáticos (EXPLOIT / NEUTRO / RIESGO) |
| game_script.py | Análisis del game script: cómo juega un equipo según el marcador |
| RankingEPAadjustado.py | Rankings ajustados por EPA y calendario |

## Semanales / temporada
| Script | Descripción |
|--------|-------------|
| DatoSemana.py | Outlier estadístico de la semana (z-score robusto) |
| MVPsSemana.py | Candidatos MVP de la semana |
| MVPsSeason.py | Seguimiento MVP de la temporada |
| resumen_partido.py | Resumen de partido: 2 PNGs — boxscore + curva Win Probability con Top-3 |WPA|, y "Claves del partido" (top-4 desviaciones de cada equipo vs su norma de temporada, con contexto del rival) |
| season_arc.py | Trayectoria de rendimiento de un equipo a lo largo de la temporada |
| Previas.py | Previews con rankings y matchups |
| proe.py | Pass Rate Over Expectation (PROE) por equipo — pass rate real menos esperado (xpass) |
| valor_turnovers.py | Valor de turnovers en EPA |

## Fórmula del campeón
| Script | Descripción |
|--------|-------------|
| contenders_tracker.py | Tracker semanal autosuficiente: descarga datos frescos y muestra qué equipos cumplen los 12 criterios de campeón hasta la semana indicada. Uso: `python contenders_tracker.py [--season N] [--week N]`. Genera contenders_s{año}_w{semana}.png |

## Draft
Éxito = 2º contrato (≥2 años) con el mismo equipo que lo drafteó, firmado 3+ años después (proxy de retención; clases 2021-2022 con ventana incompleta).

| Script | Descripción |
|--------|-------------|
| draft_success.py | Heatmap éxito por posición × ronda |
| draft_success_r1.py | Solo R1 por tramos de pick |
| draft_grid.py | Grid picks por posición × ronda |
| draft_r1_radar.py | Radar polar picks por posición 2011-2025 |
| draft_ranking_equipos.py | Ranking 32 equipos por tasa de éxito en el draft |
| draft_eval_2026.py | Evaluación Draft 2026: éxitos esperados por equipo |
| draft_value_cliff.py | Caída de éxito por ronda: small multiples 2×4 (una mini-gráfica por posición) o detalle de una posición |
| generar_dia.py | Batch: 3 PNGs por equipo → draft_calendar/ |
| generar_generales.py | 3 PNGs días generales sin inputs interactivos |
