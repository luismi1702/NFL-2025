# Calendario semanal de posts — temporada 2026

Decidido ago-2026. La maquina GENERA (semana_auto.py, dos tareas programadas
de Windows); Luis revisa y publica — la verificacion triple del CLAUDE.md
sigue siendo manual y no se automatiza.

## La semana tipo

| Dia | Post | Fuente | Generacion |
|---|---|---|---|
| Martes | Dato de la semana (outlier) | DatoSemana | batch martes 8:00 |
| Martes | Resumen del mejor partido (2 PNGs) | resumen_partido | batch martes: genera los de TODA la jornada; Luis elige cual publica |
| Miercoles | Power Rankings | power_rankings | batch martes |
| Miercoles | MVPs de la jornada | MVPsSemana (TXT, sin PNG) | batch martes |
| Jueves | Bot: balance jornada anterior + picks (gancho: previa TNF) | Manning_bot --no-retrain | batch martes (2 TXT) |
| Domingo AM | HILO de la jornada: una previa por partido, el gordo abre | Previas modo jornada | batch sabado 23:00 |
| Quincenal | Pieza tematica rotatoria (presion, PROE, rankings posicion...) | grupo B del catalogo | manual |

- Viernes y sabado sin publicar: espaciado deliberado.
- El hilo va en domingo (no viernes) para aterrizar el dia de partidos; el TNF
  ya jugado se cubre el jueves dentro del post del bot.
- Bot PUBLICO con balance honesto ("fue 11-5"): decidido ago-2026. La
  transparencia es el contenido — v6 empata con el mercado (68,9 % vs 68,2 %).

## Tareas programadas (Windows Task Scheduler, con StartWhenAvailable)

- **"NFL2025 batch martes"** — martes 8:00: `python semana_auto.py --dia martes`
  → estado_datos.txt, dato PNG, power rankings PNG, mvps_semana.txt,
  bot_balance.txt, bot_picks.txt en `salidas/{año}/w{NN}/`
- **"NFL2025 previas sabado"** — sabado 23:00: `python semana_auto.py --dia domingo`
  → un PNG por partido de la PROXIMA jornada + PDF combinado

Log de cada ejecucion: `salidas/auto_log.txt`. Un paso caido no arrastra a los
demas. Si nflverse esta caido, estado_datos.txt lo grita: NO publicar sin leerlo.

## Arranque de temporada (sem 1-3)

- Rankings, comparadores y contenders_tracker sin muestra: no publicar hasta sem 4-6.
- Bot: MIN_GAMES=2 — comprobar en sem 1-2 si saca predicciones.
- matchup_intel y piezas de personal/cobertura: pendientes de que exista
  pbp_participation_2026 (comprobar en sem 2 con estado_datos).
- Semana del kickoff: publicar el hilo pendiente docs/hilo_deberes_2026.md.

## Nivel 2 — borradores automaticos (ago-2026, EN MARCHA)

El batch del martes termina lanzando Claude Code headless (`claude -p`) con
permisos limitados a `Read/Glob/Grep/Write/WebSearch`: lee lo generado, verifica
nombres y hechos en web y escribe `salidas/{año}/w{NN}/borradores_posts.md` con
dos alternativas por post y su nota de verificacion. No publica nada.

- El prompt vive en `borradores_prompt.md`, en la raiz: editarlo ahi cambia el
  comportamiento sin tocar codigo.
- Cada corrida consume uso del plan de Claude (una sesion corta a la semana).
- `borradores_stdout.log` guarda la salida completa por si la sesion se desvia.
- Verificado con la semana 18 de 2025: escribio martes, miercoles y MVPs con
  fuentes web enlazadas, y se NEGO a redactar el post del jueves porque los
  TXT del bot no traian balance ni picks (2026 sin empezar) en vez de
  inventarlo. Ese es el comportamiento que se le pide.
- **Los borradores son borradores**: la verificacion triple del CLAUDE.md
  sigue siendo de Luis antes de publicar.

### Cierre de los posts: hashtags y menciones (decidido ago-2026)

Los ejemplos de `docs/post-ejemplos.md` son de epoca de draft
(`#NFLDraft #Equipo`) y no valen para posts semanales. La convencion de
temporada ya esta fijada y vive como regla 9 del `borradores_prompt.md`:

`#NFL | #Equipo | @cuenta1 @cuenta2`

- `#Equipo` en INGLES y sin espacios: `#Bengals`, `#49ers`, `#NYGiants`.
- Las menciones salen SOLO de `docs/cuentas-fans.md` — prohibido inventar o
  deducir un handle. Maximo 2-3 por post.
- Equipo marcado *(pendiente)* en la tabla: post sin menciones, anotado en la
  nota de verificacion.
- Piezas de liga (power rankings, MVPs, dato generico): hashtags si, menciones
  NO.
- Los caracteres se cuentan CON hashtags y menciones incluidos.

Los borradores de la w18 de 2025 son anteriores a esta regla y usan el formato
viejo (`#NFL #Bucs`, sin menciones): no sirven de muestra del formato actual.

## Nivel 2.5 — cola de copiar y pegar (ago-2026, EN MARCHA)

Decidido en vez de pagar la API de X: la maquina deja los posts listos y el
unico trabajo manual es copiar, pegar y arrastrar la imagen.

El batch del martes, tras escribir los borradores, lanza `cola_posts.py`, que
parsea `borradores_posts.md` y escribe `cola_posts.html` en la misma carpeta de
la semana. Se abre en el navegador (doble clic) y trae:

- una tarjeta por alternativa (A y B copiables por separado, la eleccion sigue
  siendo tuya),
- el PNG que acompana al post ya visible en la tarjeta, con su nombre de
  fichero para arrastrarlo desde la carpeta,
- boton **Copiar** que se lleva el texto del tuit al portapapeles,
- el conteo de caracteres RECALCULADO en Python (no el que declara el
  redactor): en verde si cabe, en rojo si se pasa de 280,
- los avisos arriba: el banner de datos sin verificar si `estado_datos.txt`
  no estaba limpio, y la lista de secciones que el redactor dejo sin escribir.

Para que esto se pueda parsear, el formato de `borradores_posts.md` es ahora un
contrato fijo (seccion `## `, linea `IMAGEN:` y cada alternativa dentro de un
bloque cercado ```post). Vive en `borradores_prompt.md`: si se toca ahi, hay
que tocar el parser de `cola_posts.py`. Si el redactor se sale del formato, la
pagina sale vacia y lo dice — no falla en silencio.

Los borradores de la w18 de 2025 son del formato viejo: no generan tarjetas.

## Nivel 3 pendiente (publicador de cola aprobada)

API de X pay-per-use ($0.015/post, sin tier gratis desde feb-2026). Solo
publicaria lo que Luis haya aprobado. No construido: decidir tras las
primeras jornadas.
