# PFF Wishlist — qué hacer cuando llegue la suscripción

Plan: PFF+ básico ($9.99/mes o $79.99/año) — incluye Premium Stats, que cubre
todo lo de abajo. Flujo de trabajo: exportar CSV desde premium.pff.com →
guardarlo en `pff_data/` → los scripts lo leen (nada de scraping).

## Pendientes

1. **Presiones cedidas por liniero individual (LT/LG/C/RG/RT)** — jul-2026.
   El motivo original de la lista: saber si al LT le ganan más que al RT.
   No existe en datos públicos (intentado con nflverse: solo se puede atribuir
   el ORIGEN del rusher, no el bloqueador batido). Con el CSV de pass blocking
   de PFF, sustituir las flechas-proxy de `oline_presion_origen.py` por los
   datos reales por puesto — el diagrama de campo ya está construido.

2. **Pressure rate real como criterio de la fórmula del campeón** — may-2026.
   Descartada del análisis original por ser propietaria (ver docs/decisiones.md).
   `discriminacion_total.py` (el script que hizo ese análisis) se eliminó
   jul-2026 — la fórmula definitiva ya vive en `contenders_tracker.py`. Con
   suscripción: replicar la validación contra los 11 campeones 2015-2025 ahí
   y decidir si entra como criterio 13.

3. **Missed tackles** — may-2026. Ídem: candidata defensiva descartada por ser
   PFF. Validar como discriminador y, aparte, posible visual de tackling por
   equipo (estilo heatmap).

## Aprobados (jul-2026) — orden de ataque al suscribirse

4. **YPRR y targets por ruta (WR/TE)** — sustituir/complementar target share
   en ranking_wrs, ranking_tes, target_share y comparadores. La métrica de
   receptores del análisis moderno.
5. **Big-time throws vs turnover-worthy plays (QBs)** — pareja narrativa para
   posts y métrica extra del comparador_qbs.
6. **Passer rating permitido y recepciones cedidas por CB/S** — convertir
   comparador_cbs y comparador_safeties en comparadores de cobertura reales
   (hoy usan stats de conteo).
7. **Pass rush win rate y tasa de dobles equipos (edges/DTs)** — 7ª métrica de
   comparador_edges y comparador_dts; el DT doblado al 70% es un post entero.
8. **Run blocking grades por hueco** — cruzar con run_gap: EPA por hueco + 
   grade del bloqueo en ese hueco. Visual único.

## Script pendiente relacionado — dline_presion_origen.py (jul-2026)

Espejo defensivo de `oline_presion_origen.py`: qué presiones GENERA cada
equipo y quién las produce — por origen (interior/exterior/blitz LB/blitz DB)
y ranking de jugadores (presiones por rusher, por snap de pass rush).

Técnicamente la v1 no requeriría PFF (atribución de sacks + QB hits del PBP,
misma técnica que el de la OL cambiando posteam por defteam), pero DECISIÓN
del usuario (jul-2026): no hacerlo a medias — se construye en agosto directamente
con los datos completos de PFF (presiones con hurries incluidos y pass rush
win rate por jugador, item 7).

## Decisión de timing

Suscribirse a finales de AGOSTO 2026 (anual): un pago cubre temporada completa
+ playoffs + primavera de draft. Verificar antes si el dato college de
prospectos requiere la suscripción college aparte.

## Reglas

- CSVs siempre descargados a mano a `pff_data/` (licencia personal, no
  commitearlos: añadir `pff_data/` a .gitignore cuando exista).
- En los posts, citar "PFF" como fuente cuando el dato venga de ahí.
