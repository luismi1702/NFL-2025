# NFL 2025 — Catálogo de scripts

## Machine Learning
| Script | Descripción |
|--------|-------------|
| Manning_bot.py | Predictor resultados v6: clasificador (XGB+LR+RF) + regresion de margen promediados, 39 features. Walk-forward 68,9% vs 68,2% del mercado (`--bench`). Flags: --retrain / --no-retrain / --week / --bench. Modelo en manning_bot_model.pkl |

## Utilidades / Cache
| Script | Descripción |
|--------|-------------|
| estado_datos.py | Semaforo de fuentes: que hay publicado, hasta que semana llega y cuanto retraso lleva. Uso: `python estado_datos.py [--season N]`. Lanzar cada martes antes de producir |
| generar_caches.py | Regenera los caches por año (fullmetrics, epa_type, situational, explosive, rz_def, newmetrics + QBs en player_stats). Uso: `python generar_caches.py 2026 [--force]` |
| server.py | Backend mínimo para ejecutar scripts desde el navegador (http://localhost:8765) |

## QBs
| Script | Descripción |
|--------|-------------|
| comparador_qbs.py | Radar dos QBs: EPA global/RZ/3er down, EPA bajo presión, **% EPA de aire** (cuánto genera el brazo y cuánto los receptores) y CPOE, + índice EPA+CPOE |
| QBsTotalEPA.py | Scatter EPA/play Red Zone vs 3er down |
| qb_presion.py | EPA bajo presión vs pocket limpio |
| qb_overview.py | Scatter todos los QBs: EPA/play vs CPOE, tamaño = intentos, color = EPA bajo presión |
| mapa_pases_por_zona.py | Mapa de pases por zona del campo |

## Ataque
| Script | Descripción |
|--------|-------------|
| comparador_wrs.py | Radar WRs: EPA/objetivo, **separación (NGS)**, YAC/recepción, aDOT, EPA en RZ y 3er down |
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
| comparador_edges.py | Radar Edge Rushers: sacks, QB hits, TFL, fumbles, EPA en sacks y **presiones/PJ reales (PFR)** |
| comparador_cbs.py | Radar de COBERTURA de dos CBs: rating permitido, % completados, yardas/objetivo, YAC/recepcion, % placajes fallados, INT/PJ. Fuente pfr_advstats; 5 de 6 metricas invertidas (menos es mejor) |
| comparador_lbs.py | Radar comparativo LBs |
| comparador_dts.py | Radar DTs: mismas 6 dimensiones que edges, con **presiones/PJ reales (PFR)** |
| comparador_safeties.py | Radar de Safeties: 4 metricas de cobertura (invertidas) + placajes/PJ y jugadas de balon/PJ. Fuente pfr_advstats |
| oline_presion.py | Tasa de presión permitida por línea ofensiva |
| oline_presion_origen.py | Por dónde cede presión cada OL. Enter = heatmap 32 (presión FTN + origen: interior DT/NT, exterior DE/OLB, blitz LB/DB, vía atribución de sacks+QB hits). Sigla = diagrama de campo con flechas de origen y los titulares de cada puesto. Nota: la presión cedida POR liniero concreto (LT vs RT) no existe en datos públicos — es charting de pago (PFF Premium Stats / SIS / FTN StatsHub) |
| dline_presion_origen.py | Espejo defensivo: desde donde GENERA presion cada defensa. Enter = heatmap 32 (presiones por 100 dropbacks, total y por origen: las cuatro columnas suman la primera). Sigla = diagrama de campo con flechas convergiendo sobre el QB (grosor = % del reparto) y el mayor generador de cada origen en su puesto. Presiones reales de pfr_advstats (con hurries); clasificacion por depth_chart_position afinada con el peso (DE >=280 lb = interior) |
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
| informe_equipo.py | Team card de un equipo: 2 PNGs (ataque y defensa). Banda superior con 3 KPIs y DONDE DOMINA en fila; columna izquierda con cada faceta en pista de ranking 1→32; franja de carrera por hueco al pie (en ataque LE->RE; en defensa el defensor que cubre cada hueco, con el orden invertido); derecha con debilidades, identidad y bloque PRESION (KPI + origen en mini-campo de 4 flechas). Defensa usa presiones reales de PFR; ataque, atribucion de sacks+QB hits |
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
