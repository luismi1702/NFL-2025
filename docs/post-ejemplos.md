# Posts @CuartayDato — Ejemplos de referencia

Este doc es la referencia de ESTILO que lee el redactor de borradores. Las
reglas mandan en `CLAUDE.md` y en `borradores_prompt.md`; aquí se ve cómo
suenan aplicadas.

## Temporada (posts semanales) — lo que se publica hoy

### Estructura

[gancho: historia, nombres propios, contexto del equipo]
→ [el dato, como revelación, no como enunciado]
→ [cierre corto con opinión]
→ `#NFL | #Equipo | @cuenta1 @cuenta2`

El cierre NO tiene por qué ser pregunta. Si la pregunta no aporta, mejor
terminar en el dato o en una frase corta con opinión.

### Ejemplos correctos

Son borradores generados y verificados para la semana 18 de 2025 (contra los
TXT del batch y contra la web). No llegaron a publicarse: valen como muestra de
tono, no como archivo de lo publicado.

---
Tampa se jugaba la temporada bajo la lluvia y la salvó su defensa: la carrera de Carolina murió en -0.57 EPA/acarreo permitido, el dato más extremo de toda la semana 18 (z=-2.65). El 16-14 se ganó en la trinchera.
#NFL | #Bucs | @Bucs_es
---
(237 caracteres) Dato de la semana. Abre con una imagen concreta (la
lluvia, el equipo jugándose el año), el dato llega en medio y el cierre es una
frase con opinión, no una pregunta.

---
Hace un año Kansas City iba a por su cuarta Super Bowl consecutiva. Hoy cierra 2025 en 6-11 y #22 del power ranking: fuera de playoffs por primera vez desde 2014 y récord perdedor por primera vez desde 2012. Las dinastías no avisan cuando se apagan.
#NFL | #Chiefs
---
(264 caracteres) Power Rankings, contado desde UN equipo. El gancho es la
caída de una dinastía, no la tabla. Cierre afirmativo.

---
Rara vez un defensa genera más EPA que nadie en toda una jornada. Devin Bush interceptó a Burrow y corrió 97 yardas para el pick six del 20-18 en Cincinnati: +12.6 EPA, por encima de las 131 yardas y 3 TD de Stevenson. El cierre de 2025 fue de la defensa.
#NFL | #Browns
---
(270 caracteres) MVPs de la jornada. Abre con lo que hace raro el
dato ("rara vez un defensa..."), no con la lista.

### Ejemplo flojo, para saber qué evitar

---
Los MVPs EPA de la semana 18: Devin Bush le devolvió a Burrow una pick six de 97 yardas (+12.6 EPA, líder de la jornada), Stevenson firmó 131 yardas y 3 TD (+11.8) y Fairbairn clavó 6/6 field goals para igualar el récord NFL de 44 en una temporada (+5.3).
#NFL
---
(260 caracteres) Mismo material, mal contado: abre con la estructura
("Los MVPs EPA de la semana 18:") y enumera tres jugadores sin jerarquía. Es un
listado, no una historia. El dato no revela nada porque nunca hubo tensión.

## Como publica Luis de verdad (leido en x.com/CuartayDato el 27-sep-2026)

Semana 2 de 2026, de la previa del TNF al duelo del sabado (hora de Madrid):

| Dia | Hora | Que |
|---|---|---|
| Jueves | ~11:30 | Previa del TNF: post + respuesta con un dato mas |
| Viernes | ~10:30 | Analisis del TNF desde UN lado (el que tiene historia) + a media tarde un post suelto de contexto (Parsons, 13:27) |
| Sabado | ~16:00 | Duelo: hilo de 3 (contexto del partido → dato del ataque → dato de la defensa rival) |
| Domingo | ~12:45 | Hilo de previas: un tuit por partido, seguidos en 5 minutos |
| Lunes | ~14:00 | Posts de partido, uno detras de otro |
| Martes | ~20:50 | Monday Night + dato de la semana |
| Miercoles | ~12:45 | Power Rankings + hilo de MVPs |

Formato que usa en TODOS (manda sobre lo que diga un prompt):
- **Parrafos cortos separados por una linea en blanco**, y otra linea en
  blanco antes de los hashtags. Nunca un bloque unico: es lo que mas cambia
  Luis al publicar un borrador (el dato del 22-sep salio del batch en un solo
  parrafo y lo publico partido en cuatro).
- El cierre de opinion va en su propia linea: `Con Shanahan a los mandos da igual quien juegue.`
- Rankings en ordinal y en prosa: `30º ataque`, `la 2ª mejor defensa`, `31º contra el pase`. Nunca `#12`.
- Decimales: en el hilo de previas los escribe con COMA (`-0,30`); en el dato
  de la semana y los posts de partido ha dejado el PUNTO del borrador
  (`+1.06`). Cualquiera vale, pero el mismo en todo el post.
- Cierre: `#NFL | #Equipo1 #Equipo2 |` y las menciones detras. En el hilo de previas, los DOS equipos y sus cuentas.
- Tuits de ~200-250 caracteres: dos datos, una frase de cierre corta (`Mal cóctel para los Colts.`, `Trece puntos y medio de línea lo dicen todo.`, `Si alguien pasa de 20, gana.`).

TODO HILO abre con una pequeña presentacion (Luis, 27-sep-2026): que se
cuenta y por que engancha, en dos o tres frases, sin tabla de numeros. Y
SIEMPRE se le dan a Luis DOS opciones de presentacion, con ganchos distintos. Asi
abren el de previas, el de MVPs (el 1/5 anuncia los cuatro nombres) y el del
duelo ("La NFL aterriza en Brasil de nuevo: Ravens-Cowboys en el Maracanã...").

Hilo de previas: ABRE CON UNA PRESENTACION sin imagen, con tres ganchos de la
jornada, y luego un tuit por partido con su PNG, seguidos en unos 10 minutos
(20-sep-2026, publicado tal cual):
```
Semana 2 en la NFL. Kansas City recibe a Indianapolis con la mejor defensa de la jornada 1, Jacksonville llega como el ataque más eficiente de la liga y Chicago viene de meter 59 puntos.

Una previa por partido, con los datos de la semana 1 🧵

#NFL
```

Dato de la semana: primera linea `Dato de la semana📕`, linea en blanco y la
historia en parrafos (22-sep-2026, publicado):
```
Dato de la semana📕

Con la mitad de los WR en la enfermería San Francisco pasó por encima de Miami: +1.06 EPA por jugada de pase.

Seattle y los Rams, segundos, con +0.60.

Con Shanahan a los mandos da igual quien juegue.

#NFL | #49ers | @49ers_Spain @LaMinaPodcast
```

Ejemplo real de un tuit del hilo de previas (20-sep-2026):
```
Pittsburgh ganó a Atlanta con el 30º ataque de la jornada (-0,30) y la 2ª mejor defensa (-0,32), 1ª contra el pase (-0,61).

El equipo más contradictorio de la semana 1 visita New England.

#NFL | #Steelers #Patriots | @cortinadeacero @PatriotsMadrid
```

Ejemplo real del analisis del TNF (25-sep-2026):
```
Llegaba con el peor ataque de la NFL en EPA por jugada y se fue de Lambeau con 502 yardas y +0,33 por jugada.

Atlanta corrió 39 veces para 244 yardas y 3 TD, con éxito en el 64% de las carreras.

El 35-14 se hizo por tierra.

#NFL | #Falcons
```

## Reglas

- No empezar con números en bruto ni con la estructura del visual.
- No centrar en el QB salvo que el visual sea de QBs.
- Cierre ilusionante, pero **prohibidas las preguntas retóricas de relleno**
  ("¿Cuánto dura la era?", "¿Quién cierra la grieta?"). Si se usa pregunta, que
  sea concreta y distinta de otras del mismo día.
- No usar "jugadores de franquicia".
- Terminología: "3er down"/"4º down" (nunca "bajada"); equipos y ciudades en
  inglés (New England, no Nueva Inglaterra); "proteger al QB"/"pass pro" (nunca
  "proteger al pasador"); "forzar turnovers" (nunca "robar balones").
- 280 caracteres contando hashtags y menciones. **El cierre cuesta**: `#NFL | #Browns` son 14 caracteres y con dos menciones se va a ~45. Un texto de 265 que cabía con los hashtags viejos hoy se pasa: recortar el texto, no el cierre.

### A quién se etiqueta

Las menciones salen SOLO de `docs/cuentas-fans.md`. Manda el TEMA DEL POST, no
el script que generó el PNG: el ejemplo de KC nace del power rankings (pieza de
liga) pero habla de un solo equipo. En una pieza de liga contada equipo a
equipo se etiqueta a ese equipo; si el texto recorre varios, hashtags y ninguna
mención.

**Se etiqueta siempre, gane o pierda el equipo** (decidido el 14-sep-2026;
antes la regla era la contraria). El ejemplo de KC de aquí abajo es anterior al
cambio y por eso va sin mención: hoy llevaría `@Chiefs_Esp @AitorManzano_`
igualmente, aunque cuente una caída. La única excepción es el equipo marcado
*(pendiente)* en la tabla, que no tiene cuenta conocida — y ahí no se inventa
un handle.

## Época de draft (abril) — otro producto

En draft el cierre SÍ era pregunta ilusionante y los hashtags eran
`#NFLDraft #Equipo`. Se conservan como referencia para la próxima ventana de
draft; **no copiar esta estructura en posts de temporada.**

---
Los Eagles ganaron la SB LIX con Hurts, Brown y Barkley. Pero la temporada 2025 acabó en WC.

Barkley, Brown y cía rondan los 30 y con poco margen salarial, el draft que los hizo grandes tiene que renovarlos. ¿Puede Roseman repetir el truco? 🤔

#NFLDraft #Eagles
@Eagles_Spain
---

---
Dan Campbell llegó y convirtió a los Lions en el mejor equipo drafteando de la NFL.
Los números lo confirman. Vamos a comprobarlo! 🤔

#NFLDraft | #Lions
@rugidos_detroit
---
