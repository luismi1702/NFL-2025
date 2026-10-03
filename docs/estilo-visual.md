# Guía de estilo visual — @CuartayDato

Versión 1 · 02-oct-2026. Nace de comparar nuestros PNG con los de SumerSports
(TNF PIT @ CLE). No es para copiarlos: es para que cada imagen nuestra se
entienda en tres segundos en un móvil.

Implementación de referencia: `lab/estilo_muestra.py`, que genera
`lab/estilo_muestra_cruce.png` (tarjeta de cruce) y
`lab/estilo_muestra_colores.png` (colores de los 32 equipos). Los scripts son
independientes (regla de CLAUDE.md): las funciones de aquí se COPIAN en cada
script, no se importan.

## 1. Principios

1. **Una imagen, una idea.** Si hacen falta dos ideas, son dos imágenes o una
   tabla de consulta (que no va a X tal cual: va al PDF).
2. **La conclusión arriba, en palabras.** El titular dice lo que el gráfico
   demuestra. El gráfico es la prueba, no el mensaje.
3. **El color significa algo o no está.** Hay dos significados y no se mezclan:
   quién es (color de equipo) y si es bueno o malo (verde y rojo, solo en los
   extremos).
4. **El número que se ve en grande, en una unidad que se siente.** %, yardas,
   puntos o jugadas. El EPA puede ser la base del cálculo y salir en pequeño;
   si sale en grande, con su media de la NFL al lado.
5. **Siempre la muestra.** Con tres partidos, "76%" sin "25 jugadas" engaña.

## 2. Tipografía

Fuentes en `fonts/` (Google Fonts, licencia OFL en `fonts/OFL.txt`). Se
registran al arrancar el script:

```python
from matplotlib import font_manager
for f in os.listdir("fonts"):
    if f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join("fonts", f))
TXT, DISP = "Barlow", "Barlow Condensed"
```

Pesos disponibles: Barlow 400/500/600/700; Barlow Condensed 600/700/800. El 800
se pide como número (`weight=800`): matplotlib no acepta `"extrabold"`.

Tres niveles, y nada más. Tamaños para un lienzo de 5,4 pulgadas de ancho a
200 dpi (1080 px):

| Nivel | Para qué | Fuente | Tamaño | Color |
|---|---|---|---|---|
| 1 · Titular | La conclusión | Barlow Condensed 700 | 22-24 | `FG` |
| 1 · Número | El dato estrella | Barlow Condensed 800 | 40-48 | `FG` |
| 2 · Etiqueta | Qué mide el número | Barlow 500, MAYÚSCULAS | 7-8 | `GRIS1` |
| 2 · Contexto | Línea bajo el titular | Barlow 400, cifras en 700 | 9,5-10 | `GRIS1`, cifras en `FG` |
| 3 · Nota | Muestra, media, definición | Barlow 400 | 6,5-7,5 | `GRIS2` |

El puesto ("1º de 32") va en Barlow 700 a 9, entre los niveles 2 y 3.

## 3. Color

Paleta base (la de CLAUDE.md, sin cambios) más dos grises de jerarquía:

```python
BG, CARD, FG, GRID, ACCENT = "#0f1115", "#151924", "#EDEDED", "#2a2f3a", "#2d6cdf"
RYG = ["#d84a4a", "#ffd166", "#06d6a0"]
GRIS1, GRIS2 = "#9aa0ab", "#5d6370"   # texto secundario / terciario
```

**Juicio (bueno o malo): RYG, solo en los extremos.** Del 1º al 5º en verde,
del 28º al 32º en rojo y todo lo demás en `GRIS1`. Un 15º no se pinta: si todo
lleva color, el color no dice nada. Se aplica al puesto, a su casilla en la
pista de 32 y, como mucho, a una barra. Nunca al fondo de una fila.

**Identidad (de quién es): el color del equipo.** Sale de
`pbp_cache/teams.parquet` con `color_equipo()` y tiene que tener un contraste
de al menos 3:1 sobre `CARD` (mínimo WCAG para gráficos y texto grande):
1. El principal, tal cual o aclarado como mucho un 20% (rojos de SF, ATL y TB;
   verde de PHI).
2. Si no llega, el color de la marca más vivo que llegue sin ser gris (naranja
   de DEN, dorado de PIT y GB, verde lima de SEA).
3. Si ninguno llega, el principal aclarado lo que haga falta.
4. Fijados a mano, porque el algoritmo elige uno que no es su identidad:
   BAL (morado aclarado un 35%) y BUF (azul aclarado un 25%).

El color de equipo se usa en el nombre del equipo, su logo, sus barras y sus
líneas. **No** en los números resaltados del texto: un "0 sacks" en el rojo de
San Francisco se lee como algo malo. Los números resaltados van en `FG` y en
negrita.

Si los dos equipos de un cruce tienen colores parecidos (KC contra TB), el
segundo usa su siguiente candidato.

## 4. Números

- **Decimales con coma en las imágenes** ("−0,13"; el signo menos es `−`, no
  `-`). El post que lleva la imagen usa la misma notación (ver
  `docs/post-ejemplos.md`: el mismo formato en todo el post).
- **Puestos en ordinal**: "7º de 32". Nunca "#7".
- **Pista de 32**: 32 casillas con la 1 a la izquierda (`pista32()`). La del
  puesto se resalta con el color de juicio (verde, rojo o `GRIS1`).
- **Muestra y referencia bajo el número**: "25 jugadas · media NFL 46%". Si
  hay dato de la temporada anterior y aporta, "2025: 41%".
- Porcentajes sin decimales salvo que el decimal cambie el orden.

## 5. Titular y contexto

- **El titular es una afirmación y se verifica como un post** (los cuatro
  puntos de CLAUDE.md).
- Una o dos líneas, unos 60 caracteres. Sin "NFL 2026", sin "X vs Y" y sin
  nombre de métrica: eso va en la etiqueta.
- Si no hay una conclusión que contar, la imagen va sin título (es de
  consulta).
- La línea de contexto (opcional) da las dos cifras que sostienen el titular.
  Se escribe con `texto_rico()`, que mide cada trozo y parte por palabras.
  Nunca colocar trozos de texto con posiciones a ojo.

## 6. Composición

- **Formato para X: 4:5** (`figsize=(5.4, 6.75)`, 1080x1350 a 200 dpi). Es lo
  que más ocupa en la pantalla del móvil sin que X lo recorte. Las tablas de
  consulta del PDF pueden ser verticales y más largas.
- Margen lateral del 7% y todo alineado a la misma izquierda.
- Orden de arriba abajo: titular, contexto, tarjeta (`CARD`, esquinas de
  0,015) con el dato, el matiz (opcional, separado por una línea `GRID`) y
  el pie.
- **No usar `bbox_inches="tight"` si hay logos con `AnnotationBbox`**: los
  descoloca. Se guarda con el lienzo fijo.
- En los cruces, ataque a la izquierda y defensa rival a la derecha, separados
  por una línea vertical `GRID` que no atraviese texto.
- **La tarjeta va con `zorder=-1`**: si no, tapa los ejes que se dibujan
  encima (`fig.add_axes`).
- **Barlow no tiene flechas** (← →): se escriben "MEJOR" y "PEOR" en los
  extremos del eje.
- **En un ranking de 32, los extremos.** Los 5 primeros y los 3 últimos con su
  logo dicen más que 32 barras. Si hace falta enseñar a todos (para situar al
  protagonista), van en gris y solo el protagonista en su color.
- Los ejemplos rehechos con piezas ya publicadas están en
  `lab/estilo_ejemplos.py` (dato de la semana, bajo centro y duelo BAL).

## 7. Pie

- Izquierda, en `GRIS2` cursiva a 6,5: las definiciones ("Blitz = 1+ blitzer
  (FTN) · Éxito = EPA > 0") y debajo el sello y la fuente ("NFL 2026 · datos
  hasta sem. 3 · nflverse PBP + FTN").
- Derecha: la marca de agua de siempre (CLAUDE.md), alineada con el sello.
- Si los datos están congelados en una semana para cuadrar con el texto, el
  sello lo dice a mano en vez de usar `sello()`.

## 8. Lista antes de dar un PNG por bueno

- [ ] ¿Se entiende la idea sin leer el post?
- [ ] ¿El titular está verificado y es el mismo dato que dice el post?
- [ ] ¿Hay como mucho un número en grande por bloque, con su muestra?
- [ ] ¿El verde y el rojo están solo en los extremos?
- [ ] ¿Ningún número resaltado va en el color del equipo?
- [ ] ¿Decimales con coma y puestos en ordinal?
- [ ] ¿Pie con definiciones, sello y marca?
- [ ] Mirarlo a tamaño de móvil (el PNG al 30%).

## Pendiente

- Aplicarla primero a `Previas.py` (en `lab/`, sin tocar producción) y después
  al resto, por orden de uso. Los titulares se estrenan cuando esté todo
  (decisión de Luis, 02-oct-2026).
- `matchup_intel.py` e `informe_equipo.py` no se pueden generar en temporada
  (necesitan `pbp_participation`, que sale en febrero).

## Migración (seguimiento)

Decidido el 02-oct-2026: los 7 scripts semanales se rediseñan en la rama
`rediseno-visual` (worktree `../NFL 2025 rediseno`, para que los batch
programados sigan usando producción) y se lanzan juntos. **El resto, poco a
poco: cada semana se migran unos cuantos**, empezando por el que toque usar.
Al migrar uno: importar `estilo.py`, titular automático solo si hay
conclusión, verificar a tamaño de móvil y marcarlo aquí.

| Script | Bloque | Estado |
|---|---|---|
| Previas.py | semanal | hecho en rama (02-oct): titular = mayor desajuste extremo; 6 cruces + FG |
| resumen_partido.py | semanal | hecho en rama (02-oct): titular = remontada, ganó con peor EPA, pérdidas o paliza; claves con nombres legibles |
| ficha_tactica.py | semanal | hecho en rama (02-oct): titular = facetas ganadas contadas por el script; colores_pareja() |
| DatoSemana.py | semanal | hecho en rama (02-oct): "Ninguna..." solo si es 1º/último en bruto; si no, "X: valor, Nº de n" |
| power_rankings.py | semanal | hecho en rama (02-oct): dos columnas de 16; titular 1º y 32º (mejorable con el movimiento semanal) |
| manning_picks.py | semanal | hecho en rama (02-oct): titular = pick más claro; naranja LED = marca del bot (excepción) |
| contenders_tracker.py | semanal | hecho en rama (02-oct): punto blanco = cumple; titular = cuántos cumplen las 12 |
| cruces.py (nuevo) | semanal | hecho (03-oct): PNG del duelo con titular automático |
| oline_presion_origen.py, dline_presion_origen.py | temática | hecho (03-oct): ficha de equipo + lista de 32, con los win rates de ESPN (datos_web/) |
| under_center.py, play_action.py, proe.py, run_gap.py, run_gap_defensa.py, oline_presion.py, qb_presion.py, qb_overview.py, QBsTotalEPA.py, cuarto_down.py, target_share.py, ranking_wrs.py, ranking_rbs.py, ranking_tes.py, RankingEPAadjustado.py, clutch_performance.py, game_script.py, series_success.py, valor_turnovers.py, mapa_pases_por_zona.py, tendencias_playcalling.py, season_arc.py, evolucion_contender.py | temática | pendiente: unos cuantos por semana |
| comparador_*.py (9), draft_*.py (7), generar_generales.py | offseason | pendiente: antes de abril |
| matchup_intel.py, informe_equipo.py, coberturas.py, red_zone_personal.py, ataque_*_personal*.py, defensa_vs_personal_ofensivo.py | sin datos hasta feb. | pendiente: en febrero |
