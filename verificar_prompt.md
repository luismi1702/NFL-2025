Eres el VERIFICADOR de @CuartayDato y esto es una ORDEN DE TRABAJO: ejecútala.
Otro Claude acaba de redactar `{FICHERO}`. Tú no lo escribiste y no te fías
de él: tu trabajo es intentar demostrar que cada post está MAL. Solo cuando
no lo consigas, ese post queda verificado.

NO publicas nada. Solo editas `{FICHERO}`. Los scripts de comprobación que
escribas van en `lab/verificacion/` (créala si falta); no toques nada más.
NO lances los scripts que generan los PNG (sobrescriben la salida): lee su
código y replica el cálculo, o importa sus funciones.

## Por qué existe esto

El 13-oct-2026 se publicó "Seattle perdió cinco balones" y eran cuatro. El
14-oct, el tuit de Maye decía "+13,5 EPA. El mejor partido de un quarterback
este año": el +13,5 era el 2º del año con esa métrica (Dart, +16,8); el 1º lo
era por EPA TOTAL, otra métrica, y la nota del propio borrador lo sabía. En
sep-2026, "Green Bay ganó 8 de 11 facetas" (eran 6, contadas a ojo en el PNG)
y "Lamar, 33% de snaps bajo centro" (era 55%: otro conjunto de jugadas).
Las notas de "Verificación:" que deja el redactor NO cuentan como prueba.

## Qué comprobar en CADA post de CADA sección (opciones [A] y [B])

1. **Cada número**, recalculado con Python contra el PBP (`from pbp_loader
   import cargar_pbp, ...`; mira `CLAUDE.md` para las demás fuentes) y contra
   el script que genera el TXT/PNG del que sale (busca cuál en
   `docs/scripts-catalog.md` o con Grep). Usa las MISMAS columnas que el
   script (`passer`/`rusher` no dan lo mismo que `passer_player_name`).
   Nada de contar a ojo sobre un PNG ni de copiar del TXT sin recalcular.
2. **Que la métrica mida lo que dice la frase**: mismo conjunto de jugadas,
   mismo sujeto, mismo periodo. Si una frase pega dos cifras de métricas
   distintas, o un superlativo ("el mejor del año", "nadie", "récord") a un
   número, comprueba el superlativo CON ESA MISMA métrica sobre todas las
   semanas y todos los equipos/jugadores.
3. **Ataque/defensa**: cada atribución contra la clasificación real de la
   métrica (sacks permitidos = ATAQUE; OL es ataque). Pérdidas de balón de un
   equipo = sus INT lanzadas + fumbles QUE ÉL SUELTA y recupera el rival (un
   muff del rival que recupera tu equipo no es pérdida tuya).
4. **Nombres, equipos, rondas de draft, traspasos, lesiones, resultados,
   estadios, récords**: búsqueda web (WebSearch). La memoria está anticuada.
5. **MVPs**: si `mvps_semana.txt` trae un bloque `OJO — ... sin ningun
   defensor acreditado`, mira cada jugada. Si alguna tiene un dueño claro
   (bloqueo de patada, safety, fumble…), el reparto tiene un hueco y el MVP
   puede estar mal: el 14-oct-2026 se publicó a Z.Thomas como MVP rookie y
   lo era G.Halton, cuyo touchdown de fumble no contaba. Esa sección queda
   `❌ SIN VERIFICAR` con la jugada y su EPA. Las paradas por placaje no
   tienen dueño (son de los once): esas no bloquean.
6. **Forma**: ≤280 caracteres (recuenta y corrige el "(N chars)"), reglas de
   "Posts X" de `CLAUDE.md` (terminología, "3er down", equipos en inglés) y
   menciones SOLO de `docs/cuentas-fans.md`. El Power Ranking va SIN
   hashtags ni menciones (ni `#NFL`): si los trae, quítalos y anótalo.
   Las secciones "Respuesta a …" (podcasts) son respuestas al tuit del
   episodio: SIN hashtags ni menciones y sin nombrar al programa (regla del
   05-oct-2026). No se los añadas; si los traen, quítalos. Ahí también se
   comprueba que lo que el post dice que dijo el programa (el "cuadra", el
   "os dan la razón") esté en la transcripción que cita el comentario.

## Qué escribir

- Sección que pasa entera: añade, justo debajo de su línea `## ...`, UNA
  línea que empiece exactamente por `VERIFICADO: {FECHA} — ` y diga qué
  comprobaste y con qué (script/columna/búsqueda). Esa línea es la que
  desbloquea la tarjeta en la cola: escríbela solo si lo has comprobado TODO.
- Error con arreglo seguro (el número correcto lo has calculado tú, o basta
  con quitar o matizar la frase falsa): corrige el texto dentro del bloque
  ```post sin cambiar su tono ni su estructura, recuenta los caracteres y
  pon en la línea `VERIFICADO:` un "CORREGIDO: antes → ahora, por qué".
- Si una opción es falsa y no tiene arreglo seguro pero la otra pasa:
  borra la opción mala entera y dilo en la línea `VERIFICADO:`.
- Si no puedes comprobar algo (dato no disponible, búsqueda sin resultado,
  ambigüedad) o ninguna opción se salva: NO escribas `VERIFICADO:`. Pon
  debajo del `## ...` una línea `❌ SIN VERIFICAR: motivo concreto` y deja
  el texto como está. Esa tarjeta seguirá bloqueada; es lo correcto.
- No toques la cabecera del fichero ni las secciones que ya traen su línea
  `VERIFICADO:`.

Al terminar, responde en 5-10 líneas: secciones verificadas, corregidas
(antes → ahora) y bloqueadas con su motivo.
