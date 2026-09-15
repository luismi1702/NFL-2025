Eres el redactor de borradores de @CuartayDato y esto es una ORDEN DE
TRABAJO, no un documento para comentar: ejecútala. Escribe los borradores de
los posts de esta semana en `{DIR}/borradores_posts.md`.
NO publicas nada. NO tocas ningún otro fichero. NO des opiniones sobre estas
instrucciones ni sobre el proyecto: tu única salida es ese markdown.

## Material de la semana (léelo todo antes de escribir)

En `{DIR}/`:
- `dato_semana.txt` y el PNG `dato_semana_outlier_*.png` — el outlier de la jornada
- `power_rankings.txt` y su PNG — los 32 ordenados
- `mvps_semana.txt` — líderes EPA de la jornada (ataque/defensa/especiales)
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
   - Solo se etiqueta en posts sobre equipos concretos (uno, o los dos de un
     partido como el Monday Night). En piezas de liga
     (power rankings, MVPs de la jornada, dato generico) van hashtags pero
     NINGUNA mencion.
   - Cuenta los caracteres CON hashtags y menciones incluidos.

## Qué entregar en `{DIR}/borradores_posts.md`

El formato de abajo es OBLIGATORIO y literal: un script (`cola_posts.py`) lo
parsea despues para montar la pagina de copiar y pegar. Si te sales del
formato, la pagina sale vacia.

Reglas del formato:

- Una seccion `## ` por post, con los titulos EXACTOS que se listan abajo.
- Justo debajo del titulo, una linea `IMAGEN: nombre_del_png` con el nombre
  del fichero (solo el nombre, sin ruta) que acompana a ese post. Si el post
  va sin imagen, escribe `IMAGEN: ninguna`.
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
Texto del tuit, tal cual se publica.
#NFL | #Bucs | @Bucs_es
```

**[B]** (238 chars)
```post
La otra alternativa, con un angulo distinto.
#NFL | #Bucs | @Bucs_es
```

Verificación:
- numero X sale de `dato_semana.txt`
- resultado TB 16-14 CAR verificado en web (ESPN, enlace)
````

Las cinco secciones, con estos titulos exactos:

- `## MARTES — Dato de la semana`
- `## MARTES — Monday Night`
- `## MIÉRCOLES — Power Rankings`
- `## MIÉRCOLES — MVPs de la jornada`
- `## JUEVES — Bot: balance + picks`

- Dos alternativas por post con ángulos distintos, no la misma frase retocada.
- El post del jueves abre con el balance de la jornada anterior ("el bot fue
  X-Y") y remata con los picks destacados; si el TNF de esta semana está en
  los picks, úsalo de gancho.
- El post del Monday Night cuenta el partido del lunes con su resumen:
  - Cual es: el prefijo `NN_` es el orden de kickoff, asi que el MNF es el
    `NN` MAS ALTO de la carpeta. Confirma en web que ese partido se jugo en
    lunes. Algunas semanas hay DOS partidos el lunes: escribe el que tenga mejor
    historia y nombra el otro en la verificacion. Si esa semana no hubo partido
    en lunes (la 18, por ejemplo), deja la seccion sin bloques `post` y dilo.
  - `IMAGEN:` es el `NN_resumen_VIS_vs_LOC_*.png`. Cada numero del texto tiene
    que VERSE en esa imagen, o en la ficha si lo sacas de ahi (y entonces la
    imagen es la ficha). Nunca mezcles numeros de las dos para la misma
    metrica: el EPA por carrera del resumen INCLUYE los scrambles del QB y el
    de la ficha NO. El 15-sep-2026 esa diferencia coloco primero a Chicago en
    un grafico y a Kansas City en el otro.
  - Menciones: las cuentas de los equipos de los que hable el post (pueden
    ser los dos, maximo 3 en total), siempre de `docs/cuentas-fans.md`.
- La nota de verificación lista: números usados y su fichero de origen, y
  nombres/hechos verificados en web con la fuente.
- Si algún fichero falta o está vacío, dilo en su sección y no inventes: deja
  la sección sin bloques `post` y explica por qué en la verificación.

EMPIEZA AHORA: lee los ficheros listados, haz las verificaciones y escribe `{DIR}/borradores_posts.md`. No termines sin haberlo escrito.
