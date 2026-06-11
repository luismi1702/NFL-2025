# Draft Content Pre-Draft 2026 — Diseño

**Fecha:** 2026-04-02
**Cuenta:** @CuartayDato
**Objetivo:** Contenido previo al Draft NFL 2026 basado en datos históricos 2011-2022

---

## Resumen

4 visualizaciones para publicar antes del Draft 2026. Las dos primeras ya existen; las dos nuevas requieren nuevos scripts.

| # | Visual | Script | Estado |
|---|--------|--------|--------|
| 1 | Éxito por posición × ronda (R1-R7) | `draft_success.py` | ✅ existe |
| 2 | Grid 32 equipos: éxito por posición y grupo de rondas | `draft_grid.py` | ✅ existe |
| 3 | Radar R1: inversión por posición por equipo | `draft_r1_radar.py` | 🆕 crear |
| 4 | Value cliff: caída de éxito por ronda por posición | `draft_value_cliff.py` | 🆕 crear |

---

## Script 3: `draft_r1_radar.py`

### Datos
- `nflreadpy.load_draft_picks(seasons=2011-2022)`
- Solo picks de ronda 1
- Agrupación por equipo y posición (POS_MAP estándar: QB, RB, WR, TE, OL, DL, LB, DB)
- **No requiere contratos OTC** — solo conteo de picks

### Modos
Script con selección de modo en runtime: `input("¿Equipo o Grid? (e/g): ")`

**Modo equipo (`e`):**
- Pide sigla del equipo: `input("Equipo (ej: SF): ")`
- Genera un radar polar de barras con 8 ejes (uno por posición)
- Longitud de barra = nº de picks R1 del equipo en esa posición (2011-2022)
- Logo del equipo centrado en el radar
- Etiqueta numérica en cada barra
- Título: `"{EQUIPO} — Inversión en R1 · 2011–2022"`

**Modo grid (`g`):**
- 32 mini-radares en cuadrícula 8×4
- Ordenados alfabéticamente por sigla (script no carga contratos OTC, no hay tasa de éxito disponible)
- Cada celda: mini-radar + logo + sigla del equipo
- Título: `"Inversión en Primera Ronda por equipo · 2011–2022"`

### Galería HTML
```
['select:Modo|equipo,grid', 'when:equipo:Equipo|SF']
```

### Estilo
- Tema oscuro estándar (BG, CARD, FG, ACCENT)
- Barras coloreadas por posición (8 colores distintos, no RYG — paleta fija por posición)
- Marca de agua `@CuartayDato` esquina inferior derecha
- DPI: 180 (equipo), 128 (grid por tamaño)
- Output equipo: `draft_r1_radar_{TEAM}.png`
- Output grid: `draft_r1_radar_grid.png`

---

## Script 4: `draft_value_cliff.py`

### Datos
- Misma lógica de éxito que `draft_success.py`:
  - `nflreadpy.load_draft_picks(seasons=2011-2022)`
  - `nflreadpy.load_contracts()` para determinar segundo contrato
  - Éxito = segundo contrato ≥2 años con mismo equipo, firmado ≥3 años después del draft
- Agrupa por posición y ronda → tasa de éxito %

### Visual
- **Line chart** horizontal (~14×8 pulgadas)
- Eje X: Rondas 1→7 (etiquetadas "R1"…"R7")
- Eje Y: % de éxito (0%–100%)
- 8 líneas (una por posición), cada una con color distinto y marcadores circulares en cada ronda
- La caída más brusca de cada posición (ronda con mayor Δ negativo) se anota automáticamente con una etiqueta flotante: `"QB ↓ aquí"`
- Grid horizontal suave (color `#2a2f3a`)
- Leyenda lateral derecha con colores por posición
- Título: `"¿A partir de qué ronda es un riesgo? — Value Cliff por posición · 2011–2022"`

### Sin modos
Siempre genera una sola imagen completa.

### Galería HTML
```
[]  # sin inputs
```

### Estilo
- Tema oscuro estándar
- Paleta de 8 colores fijos por posición (misma que `draft_r1_radar.py`)
- DPI: 180
- Output: `draft_value_cliff.png`

---

## Paleta de colores por posición (compartida entre scripts 3 y 4)

```python
POS_COLORS = {
    "QB": "#e63946",   # rojo
    "RB": "#f4a261",   # naranja
    "WR": "#2ec4b6",   # verde azulado
    "TE": "#a8dadc",   # azul claro
    "OL": "#457b9d",   # azul medio
    "DL": "#6a4c93",   # púrpura
    "LB": "#f1c453",   # amarillo
    "DB": "#06d6a0",   # verde
}
```

---

## Fuente de datos compartida
- `nflreadpy.load_draft_picks()` — picks y posiciones
- `nflreadpy.load_contracts()` — solo para scripts 1, 2 y 4 (éxito)
- Script 3 NO necesita contratos (solo conteo de picks)

## Notas de implementación
- `NICK_TO_ABBR` y `ABBR_NORM` ya definidos en los scripts existentes — copiar el mismo dict
- `normalize_team()` — misma función que en los scripts existentes
- Los scripts son independientes (no módulos compartidos, según CLAUDE.md)
