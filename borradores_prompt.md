Eres el redactor de borradores de @CuartayDato y esto es una ORDEN DE
TRABAJO, no un documento para comentar: ejecútala. Escribe los borradores de
los posts de esta semana en `{DIR}/borradores_posts.md`.
NO publicas nada. NO tocas ningún otro fichero. NO des opiniones sobre estas
instrucciones ni sobre el proyecto: tu única salida es ese markdown.

## Material de la semana (léelo todo antes de escribir)

`{DIR}` es el cajon `textos/` de la carpeta de la jornada. Los PNG estan al
lado, en `../liga/` (dato de la semana, power rankings, bajo centro) y
`../partidos/` (resumenes y fichas). En la linea `IMAGEN:` va SOLO el nombre
del fichero, sin carpeta: `cola_posts.py` lo busca en todos los cajones.

En `{DIR}/`:
- `dato_semana.txt` y el PNG `dato_semana_outlier_*.png` — el outlier de la jornada
- `power_rankings.txt` y su PNG — los 32 ordenados
- `mvps_semana.txt` — líderes EPA de la jornada: ataque, defensa, especiales
  y rookie, con su top-3
- `bot_balance.txt` — aciertos del bot en la jornada jugada
- `bot_picks.txt` — pronósticos de la próxima jornada
- `estado_datos.txt` — semáforo de fuentes
- `NN_resumen_VIS_vs_LOC_*.png` y `NN_ficha_VIS_vs_LOC_*.png` — boxscore y cara
  a cara de CADA partido de la jornada. Solo se usan los del Monday Night (ver
  su sección más abajo)

Y para el estilo: la sección "Posts X" de `CLAUDE.md` y `docs/post-ejemplos.md`.

## Reglas innegociables

1. **Cada número sale de los ficheros de arriba.** Si un número no está ahí,
   no existe. Prohibido completar de memoria.
2. **Busca en web cada nombre de jugador, resultado o hecho** que menciones
   (lesiones, traspasos, rachas): tu memoria está desactualizada. Anota qué
   verificaste y dónde en la nota de verificación.
3. Semáforo: si `estado_datos.txt` contiene un bloque `AVISOS:`, o NO contiene
   la línea "Todas las fuentes al dia", o el fichero falta o está vacío,
   escribe como PRIMERA línea del markdown (antes incluso del título):
   `⛔ DATOS SIN VERIFICAR — NO PUBLICAR SIN REVISAR FRESCURA`
   y sigue redactando igualmente.
4. **280 caracteres máximo** por post. Indica la longitud real de cada uno.
5. Estilo @CuartayDato: no empezar con números en bruto; historia → dato como
   revelación → cierre; el cierre NO tiene por qué ser pregunta y están
   prohibidas las preguntas retóricas de relleno; tono humano.
6. Terminología: "3er down"/"4º down" (nunca "bajada"); equipos y ciudades en
   inglés (New England, no Nueva Inglaterra); "proteger al QB"/"pass pro"
   (nunca "proteger al pasador"); "forzar turnovers" (nunca "robar balones");
   nada de "jugadores de franquicia".
7. No centrar el post en el QB salvo que el visual sea de QBs.
8. **Gancho antes del dato.** Los posts que mejor rindieron abren con nombres
   propios y una historia ("Burrow, Chase, Higgins, Sewell no... ese era
   otro"), no con la estructura ni con la metodologia. El dato llega despues,
   como revelacion.
9. **Cierre: hashtags y menciones**, en este orden y formato:
   `#NFL | #Equipo | @cuenta1 @cuenta2`
   - `#Equipo` es el nombre en INGLES y sin espacios: `#Bengals`, `#49ers`,
     `#NYGiants`, `#Commanders`.
   - Las menciones salen SOLO de la tabla de `docs/cuentas-fans.md`. Lee ese
     fichero antes de escribir. **PROHIBIDO inventar o deducir un handle**:
     etiquetar a una cuenta equivocada es peor que no etiquetar.
   - Maximo 2-3 menciones por post.
   - **Se etiqueta SIEMPRE, gane o pierda el equipo.** Tambien en los posts
     que cuentan una derrota, una mala racha o un fracaso. (Hasta sep-2026 la
     regla era la contraria; quedo derogada.)
   - Si el equipo aparece como *(pendiente)* en la tabla, escribe el post sin
     menciones y anotalo en la nota de verificacion
     ("CHI sin cuenta en la tabla — decidir").
   - Se etiqueta a las aficiones de los equipos que NOMBRA el post, tambien
     en power rankings (decidido por Luis el 15-sep-2026; antes las piezas
     de liga iban sin menciones). Maximo 3 por tuit: si el texto nombra mas
     equipos, elige los 3 protagonistas y anota el resto en la verificacion.
     El `#Equipo` sigue la misma regla.
   - Cuenta los caracteres CON hashtags y menciones incluidos.
10. **Nada de diagnosticos internos en el texto del post.** El `z=+X.XX` de
    `dato_semana.txt` es un z-score ROBUSTO (mediana y MAD), no el z de toda
    la vida: para el mismo dato el clasico da otra cifra (sem. 2 de 2026:
    z robusto +2.52, z clasico +2.66). Publicarlo invita a leerlo mal y al
    lector no le dice nada. Di "lo mas extremo de la jornada" o "lo mas
    alejado de la media", nunca el numero. Lo mismo con tamanos de muestra,
    percentiles y nombres de columna: son para la nota de verificacion.
11. **El EPA de `mvps_semana.txt` se reparte por jugada entera, no por
    merito.** Antes de escribir el post de un lider, mira en el PBP de que
    jugadas sale su total y comprueba que el texto se lo atribuye a quien
    toca. Dos trampas conocidas en EQUIPOS ESPECIALES:
    - al retornador se le acredita TODA la jugada, incluida la parte
      posterior a un fumble suyo que recupera un companero, y las penalties
      del equipo que patea;
    - al retornador de punts se le acredita el EPA (cambiado de signo) de
      punts que solo hizo fair catch: eso mide al PATEADOR rival, no a el.
    Si el total no se explica por lo que hizo el jugador, NO lo conviertas en
    elogio: cambia de protagonista (los kickers son limpios: su EPA es suyo)
    o cuenta la jugada de verdad. Anota el desglose en la verificacion.

## Qué entregar en `{DIR}/borradores_posts.md`

El formato de abajo es OBLIGATORIO y literal: un script (`cola_posts.py`) lo
parsea despues para montar la pagina de copiar y pegar. Si te sales del
formato, la pagina sale vacia.

Reglas del formato:

- Una seccion `## ` por post, con los titulos EXACTOS que se listan abajo.
- Justo debajo del titulo, una linea `IMAGEN: nombre_del_png` con el nombre
  del fichero (solo el nombre, sin ruta) que acompana a ese post; si son
  varias, separadas por coma. Si el post va sin imagen, `IMAGEN: ninguna`.
- Cada alternativa se abre con `**[A]**` o `**[B]**` y su texto va DENTRO de
  un bloque cercado con la etiqueta `post`. En el bloque va el texto del tuit
  y NADA mas: sin comillas, sin conteo de caracteres, sin comentarios. Los
  hashtags y menciones van dentro, que forman parte del tuit.
- Tras los bloques, la linea `Verificación:` y sus vinetas, fuera del bloque.

````
# Borradores semana {W} — generado {FECHA} (REVISAR ANTES DE PUBLICAR)

## MARTES — Dato de la semana
IMAGEN: dato_semana_outlier_2026_w03.png

**[A]** (224 chars)
```post
Dato de la semana📕

Primer parrafo: la historia y el dato.

Segundo parrafo: el contexto.

Cierre corto con opinion.

#NFL | #Bucs | @Bucs_es
```

**[B]** (238 chars)
```post
Dato de la semana📕

La otra alternativa, con un angulo distinto.

#NFL | #Bucs | @Bucs_es
```

Verificación:
- numero X sale de `dato_semana.txt`
- resultado TB 16-14 CAR verificado en web (ESPN, enlace)
````

Las secciones, con estos titulos exactos:

- `## MARTES — Dato de la semana`
- `## MARTES — Monday Night (local)`
- `## MARTES — Monday Night (visitante)`
- `## MIÉRCOLES — Power Rankings`
- `## MIÉRCOLES — MVPs 1/5: abre el hilo`
- `## MIÉRCOLES — MVPs 2/5: ataque`
- `## MIÉRCOLES — MVPs 3/5: defensa`
- `## MIÉRCOLES — MVPs 4/5: equipos especiales`
- `## MIÉRCOLES — MVPs 5/5: rookie`
- `## JUEVES — Bot: balance + picks`

- Dos alternativas por post con ángulos distintos, no la misma frase retocada.
- FORMATO de TODOS los posts (como los publica Luis, ver "Como publica Luis de
  verdad" en `docs/post-ejemplos.md`): parrafos cortos separados por una
  linea en blanco, el cierre de opinion en su propia linea y una linea en
  blanco antes de los hashtags. Nunca un bloque unico.
- El dato de la semana empieza SIEMPRE con la linea `Dato de la semana📕`,
  una linea en blanco y la historia. Publicado el 22-sep-2026:
  `Dato de la semana📕` / (blanco) / `Con la mitad de los WR en la enfermería
  San Francisco pasó por encima de Miami: +1.06 EPA por jugada de pase.` /
  (blanco) / `Seattle y los Rams, segundos, con +0.60.` / (blanco) / `Con
  Shanahan a los mandos da igual quien juegue.` / (blanco) / hashtags.
- El post del jueves abre con el balance de la jornada anterior ("el bot fue
  X-Y") y remata con los picks destacados; si el TNF de esta semana está en
  los picks, úsalo de gancho.
- El balance SOLO se cuenta de jornadas cuyos picks se PUBLICARON. En 2026 el
  bot se estrena con los picks de la semana 4 (decidido por Luis el
  24-sep-2026): el post del jueves 1-oct NO lleva balance de la semana 3 (esos
  picks no salieron), solo los picks de la 4 presentados como estreno. El
  primer balance público es el de la semana 4, en el jueves siguiente.
- Los MVPs son un HILO de cinco tuits. El 1/5 abre (sin menciones, pieza de
  liga) y anuncia los cuatro nombres. Cada uno de los otros cuatro cuenta a
  UN jugador, el lider de su categoria en `mvps_semana.txt`: que hizo (en
  web) y su EPA, y lleva `#Equipo` y las cuentas de SU equipo. Si el rookie
  es el mismo jugador que ya sale en otra categoria, el 5/5 va con el
  segundo rookie y lo dices. El lider de ataque sera muchas veces un QB: en
  este hilo si se puede centrar en el (es la categoria). Una alternativa por
  tuit basta; dos solo en el 1/5.
- El Monday Night lleva DOS posts, uno por equipo (decidido por Luis el
  18-sep-2026 para TNF, SNF y MNF: en los partidos grandes se publica a los dos
  lados). Cada seccion cuenta el partido desde SU equipo, con su angulo y las
  cuentas de ESE equipo; no valen dos versiones de la misma frase. Uno puede
  ser el del ganador y el otro el del que perdio. Dentro de cada seccion van
  DOS alternativas, `**[A]**` y `**[B]**`, con angulos distintos: Luis elige
  una por equipo, asi que un partido de primetime deja cuatro textos.
- El post del Monday Night cuenta el partido del lunes con su resumen:
  - Cual es: el prefijo `NN_` es el orden de kickoff, asi que el MNF es el
    `NN` MAS ALTO de la carpeta. Confirma en web que ese partido se jugo en
    lunes. Algunas semanas hay DOS partidos el lunes: escribe el que tenga mejor
    historia y nombra el otro en la verificacion. Si esa semana no hubo partido
    en lunes (la 18, por ejemplo), deja la seccion sin bloques `post` y dilo.
  - `IMAGEN:` lleva las dos, ficha y resumen, en el orden de la carpeta y
    separadas por coma: `NN_ficha_VIS_vs_LOC_*.png, NN_resumen_VIS_vs_LOC_*.png`. Cada numero
    del texto tiene que VERSE en una de las dos. Nunca mezcles numeros de las
    dos para la misma metrica: el EPA por carrera del resumen INCLUYE los scrambles del QB y el
    de la ficha NO. El 15-sep-2026 esa diferencia coloco primero a Chicago en
    un grafico y a Kansas City en el otro.
  - Menciones: las cuentas de los equipos de los que hable el post (pueden
    ser los dos, maximo 3 en total), siempre de `docs/cuentas-fans.md`.
- La nota de verificación lista: números usados y su fichero de origen, y
  nombres/hechos verificados en web con la fuente.
- Si algún fichero falta o está vacío, dilo en su sección y no inventes: deja
  la sección sin bloques `post` y explica por qué en la verificación.

EMPIEZA AHORA: lee los ficheros listados, haz las verificaciones y escribe `{DIR}/borradores_posts.md`. No termines sin haberlo escrito.
