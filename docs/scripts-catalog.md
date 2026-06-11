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
| comparador_qbs.py | Radar chart dos QBs: 6 métricas EPA + DAKOTA |
| QBsTotalEPA.py | Scatter EPA/play Red Zone vs 3ª bajada |
| qb_presion.py | EPA bajo presión vs pocket limpio |
| qb_overview.py | Scatter todos los QBs: EPA/play vs CPOE, tamaño = intentos, color = EPA bajo presión |
| mapa_pases_por_zona.py | Mapa de pases por zona del campo |

## Ataque
| Script | Descripción |
|--------|-------------|
| comparador_wrs.py | Radar comparativo WRs |
| comparador_rbs.py | Radar comparativo RBs |
| comparador_tes.py | Radar comparativo TEs |
| ranking_wrs.py | Scatter EPA/objetivo Red Zone vs 3ª bajada — WRs |
| ranking_rbs.py | Scatter EPA/objetivo Red Zone vs 3ª bajada — RBs |
| ranking_tes.py | Scatter EPA/objetivo Red Zone vs 3ª bajada — TEs |
| target_share.py | Target share, air yards share, WOPR, RACR por receptor |
| play_action.py | Efectividad play-action (EPA diferencial) |
| series_success.py | Heatmap 32 equipos: % posesiones por resultado |
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
| cuarto_down.py | Conversiones en 4º bajada |
| power_rankings.py | Rankings por EPA/play y calendario |
| clutch_performance.py | Rendimiento en situaciones cerradas (4Q + OT, ≤7 pts). Un equipo o scatter 32 |

## Cobertura y personal
| Script | Descripción |
|--------|-------------|
| coberturas.py | Distribución coberturas defensivas (Cover 0-6). Datos FTN 2024 |
| ataque_vs_personal_ofensivo.py | Uso y eficiencia de paquetes ofensivos propios |
| ataque_vs_personal_defensivo.py | Eficiencia ofensiva según el personal defensivo del rival |
| defensa_vs_personal_ofensivo.py | Heatmap EPA permitido según el personal ofensivo del rival |
| red_zone_personal.py | EPA en red zone según personal ofensivo. Un equipo o heatmap 32 |

## Análisis de equipo / matchup
| Script | Descripción |
|--------|-------------|
| informe_equipo.py | Informe completo de un equipo: 2 PNGs (ataque y defensa) |
| matchup_intel.py | Intel táctica A vs B — mismatches automáticos (EXPLOIT / NEUTRO / RIESGO) |
| game_script.py | Análisis del game script: cómo juega un equipo según el marcador |
| RankingEPAadjustado.py | Rankings ajustados por EPA y calendario |

## Semanales / temporada
| Script | Descripción |
|--------|-------------|
| DatoSemana.py | Outlier estadístico de la semana (z-score robusto) |
| MVPsSemana.py | Candidatos MVP de la semana |
| MVPsSeason.py | Seguimiento MVP de la temporada |
| resumen_partido.py | Resumen de partido |
| season_arc.py | Trayectoria de rendimiento de un equipo a lo largo de la temporada |
| Previas.py | Previews con rankings y matchups |
| proe.py | Probabilidad de extender una posesión |
| valor_turnovers.py | Valor de turnovers en EPA |

## Fórmula del campeón
| Script | Descripción |
|--------|-------------|
| discriminacion_total.py | Análisis definitivo: mide a los 11 campeones (2015-2025) contra ~20 métricas y valida los 12 criterios de la fórmula (sin visual, solo consola) |
| contenders_tracker.py | Tracker semanal autosuficiente: descarga datos frescos y muestra qué equipos cumplen los 12 criterios de campeón hasta la semana indicada. Uso: `python contenders_tracker.py [--season N] [--week N]`. Genera contenders_s{año}_w{semana}.png |

## Draft
| Script | Descripción |
|--------|-------------|
| draft_success.py | Heatmap éxito por posición × ronda |
| draft_success_r1.py | Solo R1 por tramos de pick |
| draft_grid.py | Grid picks por posición × ronda |
| draft_r1_radar.py | Radar polar picks por posición 2011-2025 |
| draft_ranking_equipos.py | Ranking 32 equipos por tasa de éxito en el draft |
| draft_eval_2026.py | Evaluación Draft 2026: éxitos esperados por equipo |
| draft_value_cliff.py | Caída de valor por pick en el draft |
| generar_dia.py | Batch: 3 PNGs por equipo → draft_calendar/ |
| generar_generales.py | 3 PNGs días generales sin inputs interactivos |
