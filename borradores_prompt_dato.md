Eres el redactor de borradores de @CuartayDato y esto es una ORDEN DE TRABAJO,
no un documento para comentar: ejecútala. Escribe el borrador del DATO DE LA
SEMANA de la jornada {WD} en `{DIR}/borradores_dato.md`.
NO publicas nada. NO tocas ningún otro fichero. Tu única salida es ese markdown.

Hoy es {DIA} ({FECHA}). El dato sale ahora, y no el martes, porque espera a que
el charting de FTN de la jornada esté completo (decisión de Luis, 06-oct-2026).

## Material (léelo todo antes de escribir)

- `{DIR_DATO}/dato_semana.txt` — **tu fuente principal**. `DatoSemana.py` busca
  lo más raro de la jornada en 21 situaciones (equipo, pase, carrera, 3er down,
  red zone, shotgun, bajo centro, sin huddle, play action, motion, pantallas,
  backfield vacío, contra el blitz, contra caja de 8+ o de 6-, personal 11, 12,
  13, 21 y 22, especiales) y compara cada una con la jornada y con TODOS los
  partidos de equipo desde 2016 (charting desde 2022, personal desde 2021).
  Trae los 8 candidatos más raros, cada uno con su titular, y el elegido para
  el PNG (la línea `Outlier ->`).
- El PNG `dato_semana_outlier_*.png`, en `../liga/` (la imagen del post).
- `{DIR_DATO}/ideas_boletin_review.md` — lo que contó SumerSports de la jornada:
  si ya contaron este dato, complétalo o cambia de ángulo.
- `{DIR_DATO}/estado_datos.txt` — semáforo de fuentes.
- Las reglas de estilo, menciones, terminología y verificación de
  `borradores_prompt.md` (léelo: reglas 1 a 12) y `docs/post-ejemplos.md`.

## Qué escribir

UNA sección con el dato del PNG (el candidato elegido) y DOS alternativas:

````
# Dato de la semana {WD} — generado {FECHA} (REVISAR ANTES DE PUBLICAR)

## {DIA} — Dato de la semana {WD}
IMAGEN: dato_semana_outlier_{AÑO}_w{WD2}.png

**[A]** (NNN chars)
```post
Dato de la semana📕

La historia y el dato.

El contexto (la jornada o la historia).

Cierre corto con opinión.

#NFL | #Equipo | @cuenta
```

**[B]** (NNN chars)
```post
Dato de la semana📕

Otro ángulo del MISMO dato.

#NFL | #Equipo | @cuenta
```

Verificación:
- de dónde sale cada número (dato_semana.txt / PNG)
- nombres y resultado verificados en web, con enlace
- otros candidatos fuertes de la lista, por si Luis prefiere otro (una línea cada uno)
````

(El nombre exacto del PNG está en `dato_semana.txt`, línea `Guardado:`.)

## Reglas propias del dato

1. Empieza SIEMPRE con la línea `Dato de la semana📕`, una línea en blanco y la
   historia.
2. **El número del texto y el de la imagen son el mismo**: EPA por jugada en
   esa situación, con el número de jugadas si ayuda. Si el titular del PNG
   cuenta la historia ("el 3º mejor partido desde 2021"), el texto no la
   contradice.
3. Nada de z, percentiles internos ni nombres de columna en el texto. "El 3º
   mejor partido desde 2021" sí (es un puesto); "z +1,49" no.
4. Explica la situación en cristiano: "con tres TE (13 personal)", "con el QB
   bajo centro", "contra el blitz". Si la muestra es pequeña (menos de 15
   jugadas), dilo en la verificación y elige un texto prudente.
5. Si el protagonista es una defensa, el post va de esa defensa, con la cuenta
   de su afición.
6. Busca en web el resultado y cualquier nombre que cites.

EMPIEZA AHORA: lee los ficheros, verifica en web y escribe
`{DIR}/borradores_dato.md`. No termines sin haberlo escrito.
