# Calendario semanal de posts — temporada 2026

Decidido ago-2026, ampliado con el batch del lunes en sep-2026. La maquina
GENERA (semana_auto.py, TRES tareas programadas de Windows); Luis revisa y
publica — la verificacion triple del CLAUDE.md sigue siendo manual y no se
automatiza.

## La semana tipo

| Dia | Post | Fuente | Generacion |
|---|---|---|---|
| Lunes | Un post por partido jugado (sin el MNF) | destacados + fichas | batch lunes 10:00 |
| Martes | Dato de la semana (outlier) | DatoSemana | batch martes 8:00 |
| Martes | Resumen del Monday Night (2 PNGs + ficha) | resumen_partido | batch martes: regenera TODA la jornada, que es cuando entra el MNF que el lunes faltaba |
| Miercoles | Power Rankings | power_rankings | batch martes |
| Miercoles | MVPs de la jornada: HILO de 5 (apertura + ataque, defensa, especiales, rookie) | MVPsSemana (TXT, sin PNG) | batch martes |
| Jueves | Bot: balance jornada anterior + picks (gancho: previa TNF) | Manning_bot --no-retrain | batch martes (2 TXT) |
| Viernes | Analisis del TNF jugado la noche anterior (resumen + ficha) | resumen_partido + ficha_tactica | manual (el PBP del TNF sale la misma noche) |
| Sabado | Pieza de DUELO: un partido del domingo a fondo (p.ej. Johnson vs Flores, hilo) | lab/ + PBP, FTN, participation historica | manual |
| Domingo AM | HILO de la jornada: una previa por partido, el gordo abre | Previas modo jornada | batch sabado 23:00 |
| Quincenal | Pieza tematica rotatoria (presion, PROE, rankings posicion...) | grupo B del catalogo | manual |

- Posts de partido: UNO por partido, desde el lado con mejor historia (gane o
  pierda), con resumen + ficha. Descartado el 15-sep-2026 hacer uno por equipo
  (32 a la semana, satura y la mitad serian flojos).
- EXCEPCION (18-sep-2026): los partidos de PRIMETIME (TNF, SNF, MNF) y los que
  Luis marque como grandes llevan DOS posts, uno por equipo, cada uno con su
  angulo y las cuentas de ESE equipo, y cada uno con DOS alternativas: cuatro
  textos por partido, de los que Luis publica dos. Deroga la parte del 15-sep
  que descartaba el mix de dos posts en esos partidos.
- Dentro de cada `w{N}` hay cuatro cajones (21-sep-2026), porque la carpeta
  plana mezclaba 16 previas, 45 PNG de partido, los gráficos de liga y los TXT:

  ```
  salidas/2026/w03/
    cola_posts.html    <- la página de copiar y pegar, arriba del todo
    previas/           16 PNG numerados por kickoff + previas_2026_w03.pdf
    partidos/          NN_resumen, NN_resumen_claves y NN_ficha de cada partido
    liga/              power rankings, dato de la semana, bajo centro, duelos
    textos/            borradores, destacados, estado_datos, logs
  ```

  Lo reparte `pbp_loader.salida()` por el nombre del fichero, así que ningún
  script elige carpeta a mano. En la línea `IMAGEN:` de los borradores va solo
  el nombre del PNG: `cola_posts.py` lo busca en los cajones de las DOS semanas
  y escribe la ruta relativa. El PDF de previas ya no cae en la raíz del repo.
- Cada pieza vive en la carpeta de la JORNADA DE LA QUE HABLA (20-sep-2026):
  lunes a jueves en `w{N}` (hablan de la jornada jugada) y viernes, sabado y
  domingo en `w{N+1}`, junto a los PNG que usan (analisis del TNF, pieza de
  duelo y hilo de previas). `cola_posts.py` lee las dos carpetas y monta una
  sola pagina, en la de `w{N}`.
- El viernes se abrio el 18-sep-2026 (Luis): analisis del TNF de la noche
  anterior, con los PNG de resumen_partido y ficha_tactica del partido. El PBP
  del jueves esta publicado el viernes por la manana; FTN todavia no, asi que
  nada de play action ni blitz en esa pieza. El TNF deja de cubrirse dentro del
  post del bot del jueves: el jueves va la PREVIA, el viernes el analisis.
- El sabado se abrio el 17-sep-2026
  (Luis) para generar contenido tambien ese dia: una pieza de duelo de UN solo
  partido del domingo, mas profunda que la previa. Para no solaparse con el
  hilo del domingo, ese partido NO abre el hilo de previas.
- Hasta que el bot tenga muestra (semana 3), el jueves va la previa del TNF sola
  (hecho el 17-sep-2026 con DET @ BUF).
- El hilo va en domingo (no viernes) para aterrizar el dia de partidos.
- Bot PUBLICO con balance honesto ("fue 11-5"): decidido ago-2026. La
  transparencia es el contenido — v6 empata con el mercado (68,9 % vs 68,2 %).

## Tareas programadas (Windows Task Scheduler, con StartWhenAvailable)

- **"NFL2025 batch lunes"** — lunes **10:00**: `python semana_auto.py --dia lunes`
  → resumenes y fichas tacticas de TODOS los partidos del domingo, bajo centro,
  `destacados.txt` (rastreo de lo que merece post) y un borrador por partido
  con `borradores_prompt_lunes.md` en `borradores_lunes.md`, mas su cola HTML.
  El Monday Night NO esta: se juega esa noche y lo recoge el batch del martes.
  A las 10:00 y no a las 8:00 porque el Sunday Night acaba sobre las 5:30 hora
  espanola y nflverse puede tardar en publicarlo.
- **"NFL2025 batch martes"** — martes 8:00: `python semana_auto.py --dia martes`
  → estado_datos.txt, dato PNG, power rankings PNG, mvps_semana.txt,
  bot_balance.txt, bot_picks.txt en `salidas/{año}/w{NN}/`
- **"NFL2025 previas sabado"** — sabado 23:00: `python semana_auto.py --dia domingo`
  → un PNG por partido de la PROXIMA jornada + PDF combinado

- **"NFL2025 recuperar batch"** — al iniciar sesion (+3 min):
  `python semana_auto.py --recuperar` → si el ULTIMO batch programado no llego
  a arrancar, lo lanza; si arranco, no hace nada. Solo el ultimo: el martes
  regenera lo del lunes y relanzar uno viejo pisaria `borradores_posts.md`.

Por que existe (15-sep-2026): las tareas son de tipo *Interactive* y solo
corren con sesion iniciada. Windows Update reinicio a las 2:29, la sesion no se
abrio hasta las 10:47 y el batch del martes se salto sin dejar rastro;
`StartWhenAvailable` solo recupera si el equipo estaba APAGADO. Pasarlas a
"ejecutar aunque no haya sesion" (S4U) necesita admin. Ese dia tambien se quito
"no iniciar con bateria" de las tres tareas.

Log de cada ejecucion: `salidas/auto_log.txt`. Un paso caido no arrastra a los
demas. Si nflverse esta caido, estado_datos.txt lo grita: NO publicar sin leerlo.

## Arranque de temporada (sem 1-3)

- Rankings, comparadores y contenders_tracker sin muestra: no publicar hasta sem 4-6.
- Bot: MIN_GAMES=2 — comprobado en 2026: NO hay post del bot los jueves de
  las semanas 2 y 3 porque no hay picks. Los primeros son los de la semana 3,
  generados en el batch del martes tras la semana 2. El primer balance llega el
  martes siguiente a la semana 3. Sin muestra, el bot sale con codigo 3 y el log dice
  SIN MUESTRA, no FALLO.
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
- Desde el 15-sep-2026 tambien redacta el post del Monday Night (seccion
  `## MARTES — Monday Night`). Antes caia entre los dos prompts: el del lunes
  lo excluye porque aun no se ha jugado y el del martes no lo pedia.
- El prompt se pasa por STDIN. Como argumento, `claude.CMD` (cmd.exe) lo
  cortaba en el primer salto de linea: por eso fallaron el 08-sep y el 15-sep
  con un "tu mensaje se corto".
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
- Se etiqueta a las aficiones de los equipos que NOMBRA el post, tambien en
  power rankings (decidido por Luis el 15-sep-2026; antes las piezas de liga
  iban sin menciones). Maximo 3 por tuit. Los MVPs van en hilo para que cada
  jugador lleve las cuentas de su equipo; la apertura, sin menciones.
- Los caracteres se cuentan CON hashtags y menciones incluidos.

Los borradores de la w18 de 2025 son anteriores a esta regla y usan el formato
viejo (`#NFL #Bucs`, sin menciones): no sirven de muestra del formato actual.

## Borradores del lunes y del martes: ficheros separados (15-sep-2026)

- El lunes escribe `borradores_lunes.md` (un post por partido) y el martes
  `borradores_posts.md` (dato, MNF, power rankings, hilo de MVPs, bot).
  Compartian fichero y el batch del martes borraba los del lunes.
- `cola_posts.py` junta los dos en una sola pagina AGRUPADA POR DIA de
  publicacion, con un rotulo por dia; dentro de cada dia, en el orden de la
  carpeta (por nombre de fichero, tambien las imagenes de cada post). Todo
  partido de la carpeta tiene tarjeta, aunque no tenga post.
- Timeout del redactor: 1 hora, porque ya son 15 posts con verificacion web.

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
