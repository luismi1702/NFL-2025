---
name: publicar
description: Publica en X (@CuartayDato) los posts de la cola del día, desde el Chrome de Luis. Úsala cuando Luis diga "/publicar", "publica lo de hoy", "sube el hilo" o pida publicar los posts de un día concreto. Vale para cualquier día de la semana.
---

# /publicar — publicar la cola del día en X

Publicas tú, en el navegador de Luis (Claude in Chrome), los posts que ya están
redactados en la cola. No redactas aquí: eso es trabajo de los borradores.
Argumento opcional: el día (`/publicar martes`). Sin argumento, hoy.

**Luis lo sigue DESDE EL MÓVIL** (14-oct-2026: "lo que quiero es no tener que
estar en el ordenador"). Todo lo que tenga que VER para decidir (candidatas de
fotos, el compositor antes de publicar, el hilo publicado) le llega con
`SendUserFile` (`display: "render"`). Nunca "lo tienes en tu Chrome" ni abrir
cosas en su PC. Mensajes cortos: qué va a salir y la pregunta.

## Reglas que no se saltan

1. **Visto bueno de Luis antes de cada tanda.** Enseñas qué va a salir (texto
   exacto, imágenes, orden, qué es hilo y qué va suelto) y esperas su "sí". El
   visto bueno de una tanda no vale para otra ni para otro día. Un hilo entero
   es UNA tanda: con su sí se publica seguido.
2. **El texto sale tal cual está en la cola.** Si Luis cambia algo (le pasó con
   el Power Ranking del 14-oct: tres versiones del cierre en dos minutos), el
   cambio se hace PRIMERO en el borrador (`salidas/{año}/wNN/textos/borradores_*.md`,
   recontando el "(N chars)"), se regenera la cola (`python cola_posts.py`) y
   se comprueba con `publicar_cola.py` que la tarjeta sigue desbloqueada.
   Después se rehace el compositor. Cola y publicado tienen que coincidir.
3. **Nunca se publica** una tarjeta con `bloqueado`, con `faltan_imagenes`,
   con `se_pasa` (más de 280), ni las `manual` (respuestas a podcasts: esas las
   pone Luis). `bloqueado` es "Preview"/"Review" (falta el boletín de
   SumerSports) o "sin verificar": nadie ha pasado sus cifras por el PBP. Desde
   el 14-oct-2026 eso lo hace el batch (`semana_auto.verificar()`); si aun así
   sale bloqueada, se verifica (`python semana_auto.py --verificar <borrador>`
   o a mano), se apunta `VERIFICADO: fecha — qué se comprobó` debajo del `## `
   de ESA sección (o en la cabecera, si se ha revisado el fichero entero) y se
   relanza `cola_posts.py`. Nunca en la cabecera si hay secciones sin revisar:
   desbloquearía también las del jueves, que se publican otro día.
4. **EN UN HILO, CADA RESPUESTA SALE SIN MENCIONAR A LAS CUENTAS DEL POST
   ANTERIOR.** Es lo más importante de esta skill (Luis, 12-oct-2026). Ver
   "Respuesta dentro de un hilo". Si no puedes comprobar que las casillas
   están desmarcadas, NO publicas esa respuesta: paras y avisas.
5. Ante un inicio de sesión, un captcha, un aviso de X o cualquier diálogo que
   no esperabas: paras y avisas. No inicias sesión, no aceptas condiciones.
6. Si algo falla a mitad, **no reintentes a ciegas**: mira primero en el perfil
   o en el hilo si el post salió (recarga y espera 3-4 s: X tarda en
   mostrarlo). Un duplicado es peor que un hueco.

## 1. Qué toca

```
python publicar_cola.py              # hoy
python publicar_cola.py --dia martes # otro día
```

Devuelve JSON: `week`, `dia`, `aviso`, `sin_verificar` y `posts` en el orden de
publicación. De cada post: `titulo`, `imagenes` (rutas absolutas, ya en el
orden en que se suben), `posts` (`letra`, `texto`, `caracteres`), `publicado`
(si ya salió, con su URL), `bloqueado`, `manual`, `hilo`.

- `aviso` "esto es de la wNN anterior" es lo normal de lunes a miércoles: la
  semana de la cola va de jueves a miércoles. Díselo en una línea y sigue.
- `sin_verificar: true` solo dice que el fichero lleva el banner ⛔; lo que
  manda es `bloqueado` de cada tarjeta. Si ninguna está bloqueada, basta con
  mencionarlo.
- Lo que tiene `publicado` se salta.

## 2. Qué es hilo y qué va suelto

- Una sección con "Hilo" y "presentación" (o "abre el hilo") en el título abre
  un hilo. Las numeradas `1/5`, `2/5`… son ese hilo, en ese orden.
- **Lunes:** el hilo lo abre "Hilo de la jornada: presentación" y cuelgan de él
  todos los posts de partido (los que llevan `NN_1_resumen` y `NN_2_ficha`).
  La "Previa del Monday Night" va suelta.
- **Domingo:** "Hilo de previas 0/N" abre y cuelgan las previas en orden.
- Dato de la semana, Power Rankings, bot y previas sueltas: posts independientes.
- El orden del JSON mezcla sueltos e hilo (el miércoles sale MVPs 1/5, PR,
  MVPs 2/5…): pregunta a Luis qué va primero. El 14-oct eligió PR primero y
  luego el hilo entero.
- Si dudas de si algo cuelga del hilo o va suelto, pregúntalo.

Dentro del hilo cada tuit responde AL ANTERIOR (cadena), no todos al primero.

### Qué imágenes y etiquetas lleva cada cosa

Así publica Luis (visto en su perfil el 12-oct-2026, semanas 2 a 5). Si lo que
trae el JSON no cuadra con esta tabla, se pregunta antes de publicar.

| Pieza | Estructura | Imágenes | Hashtags y menciones |
|---|---|---|---|
| Lunes: hilo de la jornada | presentación + un tuit por partido, en cadena | presentación, ninguna; cada partido, DOS: `NN_1_resumen` y `NN_2_ficha`, en ese orden | los del borrador |
| Domingo: hilo de previas | presentación + una previa por partido | presentación, ninguna; cada previa, su `NN_preview` (y desde el 11-oct, detrás, el `NN_preview_..._claves`) | los del borrador |
| Previa suelta (TNF, MNF) | un post | igual que una previa del hilo | los del borrador |
| Sábado: duelo | hilo | los primeros tuits, un PNG cada uno; los últimos, sin imagen | los del borrador |
| Miércoles: MVPs | presentación + ataque, defensa, especiales, rookie | presentación, ninguna; cada MVP, una FOTO del jugador en ese partido (ver "Fotos de los MVPs") | los del borrador (cada MVP, su afición) |
| Jueves: bot | post + respuesta "Cómo funciona" | el post, `manning_picks`; la respuesta, ninguna | solo `#NFL`: **ni menciones ni `#Equipo`** (Luis, 15-oct-2026). Si el borrador las trae, se quitan antes (regla 2) |
| Dato de la semana | un post | su PNG | los del borrador |
| **Power Rankings** | un post | su PNG | **NINGUNO, ni `#NFL`** (Luis, 14-oct-2026). Si el borrador los trae, se quitan en el borrador antes (regla 2) |
| Respuestas a podcasts u otras cuentas | — | no se publican desde aquí | — |

### Fotos de los MVPs

Luis lo hace así y así se hace aquí (12-oct-2026): Google Imágenes con
"<jugador> vs <rival>" (el rival de ESE partido; sale del texto del tuit o de
`MVPsSemana`), y se elige una foto del partido. Prepáralas mientras Luis elige
las opciones [A]/[B], para no hacerle esperar.

1. Una búsqueda por MVP, filtrada a la última semana:
   `https://www.google.com/search?q=<jugador>+<rival>&tbm=isch&tbs=qdr:w`
   (para un defensa, "+interception"; para un kicker, "+field+goal").
2. Dos o tres candidatas por jugador. Abre la vista previa (clic en la
   miniatura; si el panel no sale, otro clic) y comprueba con el pie de foto
   y la imagen: ese jugador (dorsal, nombre), ese partido (rival, fecha; el
   nombre del fichero a veces lo dice: `CTC-L-BEARS-PACKERS-101126`). Lo que
   no puedas confirmar, dilo. Descarta las que llevan marca de agua ("Connect
   partner" de Reuters) o un rótulo de vídeo ("HIGHLIGHTS").
3. URL original y resolución: con la vista previa abierta, espera 2-3 s y lee
   con `javascript_tool`:
   ```js
   [...document.querySelectorAll('img')].filter(i=>i.src.startsWith('http') && !i.src.includes('gstatic') && i.naturalWidth>300).map(i=>{const u=new URL(i.src);return [u.origin+u.pathname, i.naturalWidth+'x'+i.naturalHeight]})
   ```
   Si sale vacío, espera y repite; si sigue vacío, esa web no deja (descártala).
   Saca la versión grande quitando el redimensionado: el proxy de Yahoo
   (`s.yimg.com/.../https%3A%2F%2Fmedia.zenfs.com...`) es la URL de
   `media.zenfs.com`; `i0.wp.com/<web>/...` es `<web>/...`; Cloudinary del
   NFL (`static.clubs.nfl.com/image/upload/t_..._mobile/...`) sin el `t_...`;
   minutemedia (SI), cambiando `w_720` por `w_1600`.
4. Bájalas con `curl -sL -A "Mozilla/5.0"` a una carpeta del scratchpad y
   monta UNA hoja con PIL (fondo `#0f1115`, cada candidata con su etiqueta
   "MAYE 1 · ESPN · 1920x1080" en amarillo). Mándala con `SendUserFile` y pide
   la elección en una línea ("Maye 1, Green 2…").
5. Las elegidas se copian a `salidas/{año}/wNN/liga/mvp_fotos/` con nombre
   claro (`mvp_ataque_maye.jpg`, `mvp_defensa_green.jpg`…). Solo imágenes.
6. Se suben con `file_upload` en el tuit de ese MVP.

Si no aparece una foto clara de ese partido, no se pone una de otro: se le
dice a Luis y ese tuit lo decide él. Si solo hay una a baja resolución (Myers,
912x513, el 14-oct), dilo junto a la hoja.

## 3. Elegir [A] o [B] y pedir el visto bueno

Enseña la lista numerada con las dos opciones de cada post (la primera frase
de cada una basta) y pregunta en una línea: qué letra, y en qué orden. Luis
contesta en corto ("1B y 2B, 3 primero el power ranking"). Después monta el
primer post en el compositor, manda la captura (`zoom` con
`save_to_disk: true` del cuadro → `SendUserFile`) con el resumen de lo que
sigue, y espera el sí. Los hilos se publican seguidos; si Luis quiere parar en
medio, se para y se retoma con `publicado`.

## 4. Navegador

Carga las herramientas en UNA llamada a ToolSearch: tabs_context_mcp,
tabs_create_mcp, navigate, computer, find, read_page, file_upload,
javascript_tool, tabs_close_mcp (y `SendUserFile` si no está).
`tabs_context_mcp` primero; pestaña nueva o la vacía del grupo; `x.com/home`.
Comprueba que la sesión abierta es **@CuartayDato** (abajo a la izquierda). Si
es otra o no hay sesión, para. Chrome no abre `file:///`: lo local se le
manda a Luis con `SendUserFile`, nunca con el navegador.

## 5. Post suelto (o el primero de un hilo)

1. Abre `https://x.com/compose/post` y espera 2 s. `find` "Texto del post".
2. Clic en el cuadro y escribe el texto con `type` (los saltos de párrafo
   salen bien).
3. **Desplegable de sugerencias:** si el texto acaba en una @mención o un
   #hashtag, queda abierto. Se cierra SIN elegir nada haciendo clic en una
   zona vacía del texto, a la derecha de la última línea (nunca Escape: puede
   cerrar el compositor; nunca clic en el desplegable). No escribas un espacio
   final para cerrarlo: al borrarlo se vuelve a abrir.
4. Imágenes: `find` "file input de imágenes del compositor" y `file_upload`
   con todas, en una llamada y en el orden del JSON. Acepta las rutas de
   `salidas/` y del scratchpad. No hagas clic en el botón de la imagen: abre
   un diálogo del sistema que no ves. Espera 2-3 s a que cargue (hasta
   entonces "Postear" sale gris).
5. Para rehacer el texto (Luis cambia algo): clic en el cuadro, `ctrl+a` y
   volver a escribir. La imagen se queda.
6. **Comprueba antes de publicar**, con captura: texto idéntico al de la cola,
   menciones y hashtags enteros, desplegable cerrado, imágenes cargadas y en
   orden, contador sin rojo (el círculo muestra lo que QUEDA: 280 − lo que
   marca = longitud; tiene que cuadrar con `caracteres`).
7. Clic en "Postear" (`find` "Postear button in compose dialog"). El
   compositor se cierra y vuelve a /home. Espera 3 s antes de navegar: si no,
   salta un "Leave site?" y la navegación se bloquea (no fuerces: espera y
   mira el perfil).
8. URL: abre `x.com/CuartayDato`, espera 3-4 s (a veces carga con error la
   primera vez: recarga) y lee el id del primer post:
   ```js
   document.querySelector('article a[href*="/status/"] time').closest('a').getAttribute('href').split('/').pop()
   ```
   Comprueba con `innerText` que ese article es el post que acabas de
   publicar (y no uno fijado o anterior).
9. Apúntalo:
   `python publicar_cola.py --dia <dia> --marcar "<titulo exacto>" --letra A --url https://x.com/CuartayDato/status/<id>`

## 6. Respuesta dentro de un hilo

1. Abre la URL del tuit anterior del hilo y espera 3 s.
2. `find` "reply button of the focused post '<primeras palabras>'" y clic.
3. Mira la línea encima del cuadro. Según lo que mencionaba el tuit anterior:
   - **Nadie** (la presentación sin menciones): el compositor dice "Añadir
     otro post" y NO hay línea "Respondiendo a". Es correcto: no hay nada que
     desmarcar.
   - **Alguna cuenta**: dice "Respondiendo a @CuartayDato y a @X" o "En
     respuesta a @CuartayDato @X y @Y". Clic en esa línea (los nombres en
     azul): se abre el diálogo "Respondiendo a", con @CuartayDato arriba (fija)
     y debajo "Otros en esta conversación". La lista tarda ~2 s en cargar.
     - **Una sola cuenta**: no hay casilla general; desmarca la de la cuenta.
     - **Dos o más**: hay casilla general en "Otros en esta conversación";
       desmárcala y se desmarcan todas.
     Comprueba con `zoom` que no queda NINGUNA marcada salvo @CuartayDato y
     pulsa "Listo" (arriba a la derecha).
4. De vuelta en el compositor, `zoom` a la línea: tiene que decir SOLO
   "Respondiendo a @CuartayDato". Si no consigues dejarla así o no puedes
   leerla con seguridad: NO publiques. Para y avisa con la captura.
5. Clic en el cuadro ("Postea tu respuesta"), escribe, cierra el desplegable
   (paso 5.3), sube la imagen (paso 5.4; cada respuesta tiene su propio file
   input: vuelve a hacer `find`).
6. **Vuelve a comprobar la línea "Respondiendo a @CuartayDato"** con captura
   justo antes de pulsar "Responder". Es la única regla que no admite fallo.
7. Publica, espera 4 s, abre el tuit ANTERIOR (no el perfil) y lee el id de la
   respuesta nueva, que es el último `article` del hilo:
   ```js
   [...document.querySelectorAll('article')].map(a=>{const t=a.querySelector('a[href*="/status/"] time');return [t?t.closest('a').getAttribute('href').split('/').pop():'(foco)', a.innerText.replace(/\n/g,' ').slice(0,70)]})
   ```
   (el tuit en foco sale sin `time` cuando es el de la URL: su id ya lo
   tienes). Apúntalo con `--marcar`. La siguiente responde a esta.

Las menciones que van escritas en el texto del propio post (la afición del
equipo) sí salen: lo que se quita son las heredadas del post anterior.

## 7. Al terminar

- Captura del hilo o del post publicado (`zoom` de la columna central,
  `save_to_disk`) → `SendUserFile` con el enlace en el pie.
- `publicar_cola.py --dia <dia>`: comprueba que todo lo publicado tiene su
  `publicado` con URL.
- Resume en una tabla qué salió (con enlaces), qué se saltó y por qué, y qué
  queda pendiente. Si paraste a mitad, di exactamente cuál fue el último tuit
  publicado.
- Cierra la pestaña.
- Lo nuevo que hayas aprendido de la página de X va a "Notas de campo".

## Notas de campo

Lo que se vaya aprendiendo de la página de X se apunta aquí, con fecha. Lo ya
incorporado a los pasos de arriba no se repite.

- 14-oct-2026, estreno: Power Ranking w05 + hilo de MVPs (5 tuits), todo bien
  a la primera; 15 min de punta a punta, la mitad en las fotos. Lo aprendido
  está en los pasos 2, 3, 5 y 6.
- 16-oct-2026, hilo Jaguars: la presentación salió con "@JaguarsESPsi". Entre
  la captura del visto bueno y el clic se coló un "si" en el compositor
  abierto (con el foco en el texto), y el primer clic en "Postear" no publicó.
  Luis lo borró y se republicó. Regla: JUSTO antes de cada clic en
  "Postear"/"Responder" (también al reintentar), leer el texto con
  `javascript_tool` (`[role="dialog"] [data-testid="tweetTextarea_0"]`,
  cambiando `\n\n\n` por `\n\n`) y compararlo con la cola carácter a
  carácter. No dejar el compositor con el foco esperando el sí: montarlo,
  capturar y, si tarda, recomprobar.
- En un hilo propio, la primera respuesta a un tuit con mención sale como
  "Añadir otro post" sin línea "Respondiendo a": X no hereda la mención. Las
  siguientes dicen "Respondiendo a @CuartayDato". En ambos casos no hay nada
  que desmarcar.
- El contador de X cuenta "−" (signo menos, U+2212) como 2: un 261 de la cola
  sale como 262. No es un error del texto.
- 17-oct-2026, hilo CHI @ ATL (7 tuits): en el 6/7, un clic en el cuadro de
  respuesta por `ref` (de `find`) cayó en el botón de programar
  (`/compose/post/schedule`) y el compositor volvió vacío como post nuevo. No
  salió nada, pero hubo que rehacer la respuesta. Clic al cuadro por
  coordenadas sacadas de la captura, y antes de publicar, la comprobación JS
  de una vez: texto, línea "Respondiendo a", nº de imágenes
  (`[data-testid="attachments"] img`) y que el botón diga "Responder" (si
  dice "Postear", no es una respuesta). Publicar con
  `document.querySelector('[role="dialog"] [data-testid="tweetButton"]').click()`
  funciona y evita fallar el botón.
- 18-oct-2026, hilo de previas w06 (13 tuits, ~20 min): todo a la primera.
  · En la primera respuesta ("Añadir otro post", sin línea "Respondiendo a")
    el botón dice "Postear" y ES una respuesta (el compositor muestra el
    tuit anterior encima con la línea del hilo). La regla "si dice Postear,
    no es respuesta" vale solo cuando hay línea "Respondiendo a".
  · Abrir la respuesta sin fallar: `[...document.querySelectorAll('article')]
    .find(a=>a.innerText.includes('<inicio del tuit>'))
    .querySelector('[data-testid="reply"]').click()`.
  · Desmarcar en el diálogo "Respondiendo a" por JS funciona: si hay casilla
    general (un `input[type=checkbox]` no deshabilitado fuera de
    `[data-testid="UserCell"]`) y está marcada, `.click()`; si no, click en
    cada una marcada y no deshabilitada. Leer luego `checked` de todas.
  · Cerrar el desplegable de @menciones cuando la última línea llena el
    ancho: clic en la línea en blanco entre el último párrafo y los hashtags.
