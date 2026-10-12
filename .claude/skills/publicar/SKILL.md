---
name: publicar
description: Publica en X (@CuartayDato) los posts de la cola del día, desde el Chrome de Luis. Úsala cuando Luis diga "/publicar", "publica lo de hoy", "sube el hilo" o pida publicar los posts de un día concreto. Vale para cualquier día de la semana.
---

# /publicar — publicar la cola del día en X

Publicas tú, en el navegador de Luis (Claude in Chrome), los posts que ya están
redactados en la cola. No redactas ni corriges aquí: eso es trabajo de los
borradores. Argumento opcional: el día (`/publicar martes`). Sin argumento, hoy.

## Reglas que no se saltan

1. **Visto bueno de Luis antes de cada tanda.** Enseñas qué va a salir (texto
   exacto, imágenes, orden, qué es hilo y qué va suelto) y esperas su "sí". El
   visto bueno de una tanda no vale para otra ni para otro día.
2. **El texto sale tal cual está en la cola.** Si algo hay que cambiar, se lo
   dices a Luis, se cambia en el borrador y se vuelve a listar.
3. **Nunca se publica** una tarjeta con `bloqueado` (falta el boletín de
   SumerSports), con `faltan_imagenes`, con `se_pasa` (más de 280), ni las
   `manual` (respuestas a podcasts: esas las pone Luis).
4. **EN UN HILO, CADA RESPUESTA SALE SIN MENCIONAR A LAS CUENTAS DEL POST
   ANTERIOR.** Es lo más importante de esta skill (Luis, 12-oct-2026). Ver
   "Respuesta dentro de un hilo". Si no puedes comprobar que la casilla está
   desmarcada, NO publicas esa respuesta: paras y avisas.
5. Ante un inicio de sesión, un captcha, un aviso de X o cualquier diálogo que
   no esperabas: paras y avisas. No inicias sesión, no aceptas condiciones.
6. Si algo falla a mitad, **no reintentes a ciegas**: mira primero en el perfil
   si el post salió. Un duplicado es peor que un hueco.

## 1. Qué toca

```
python publicar_cola.py              # hoy
python publicar_cola.py --dia martes # otro día
```

Devuelve JSON: `week`, `dia`, `aviso`, `sin_verificar` y `posts` en el orden de
publicación. De cada post: `titulo`, `imagenes` (rutas absolutas, ya en el
orden en que se suben), `posts` (`letra`, `texto`, `caracteres`), `publicado`
(si ya salió, con su URL), `bloqueado`, `manual`, `hilo`.

- Si `aviso` no está vacío (la carpeta nueva no tiene nada y se ha tirado de la
  anterior), díselo a Luis antes de seguir: puede ser material de la semana pasada.
- Si `sin_verificar` es true, recuérdale que el borrador lleva el aviso de
  datos sin verificar y pregúntale si ya lo ha revisado.
- Lo que tiene `publicado` se salta.

## 2. Qué es hilo y qué va suelto

- Una sección con "Hilo" y "presentación" en el título abre un hilo. Las
  numeradas `1/5`, `2/5`… son ese hilo, en ese orden.
- **Lunes:** el hilo lo abre "Hilo de la jornada: presentación" y cuelgan de él
  todos los posts de partido (los que llevan `NN_1_resumen` y `NN_2_ficha`).
  La "Previa del Monday Night" va suelta.
- **Domingo:** "Hilo de previas 0/N" abre y cuelgan las previas en orden.
- Dato de la semana, Power Rankings, bot y previas sueltas: posts independientes.
- Si dudas de si algo cuelga del hilo o va suelto, pregúntalo.

Dentro del hilo cada tuit responde AL ANTERIOR (cadena), no todos al primero.

### Qué imágenes lleva cada cosa

Así publica Luis (visto en su perfil el 12-oct-2026, semanas 2 a 5). Si lo que
trae el JSON no cuadra con esta tabla, se pregunta antes de publicar.

| Pieza | Estructura | Imágenes |
|---|---|---|
| Lunes: hilo de la jornada | presentación + un tuit por partido, en cadena | presentación, ninguna; cada partido, DOS: `NN_1_resumen` y `NN_2_ficha`, en ese orden |
| Domingo: hilo de previas | presentación + una previa por partido | presentación, ninguna; cada previa, su `NN_preview` (y desde el 11-oct, detrás, el `NN_preview_..._claves`) |
| Previa suelta (TNF, MNF) | un post | igual que una previa del hilo |
| Sábado: duelo | hilo | los primeros tuits, un PNG cada uno; los últimos, sin imagen |
| Miércoles: MVPs | presentación + ataque, defensa, especiales, rookie | presentación, ninguna; cada MVP lleva una FOTO del jugador en ese partido, que no es un PNG nuestro: se busca como en "Fotos de los MVPs" |
| Jueves: bot | post + respuesta "Cómo funciona" | el post, `manning_picks`; la respuesta, ninguna |
| Dato de la semana, Power Rankings | un post | su PNG |
| Respuestas a podcasts u otras cuentas | — | no se publican desde aquí |

### Fotos de los MVPs

Luis lo hace así y así se hace aquí (12-oct-2026): Google Imágenes con
"<jugador> vs <rival>" (el rival de ESE partido; sale del texto del tuit o de
`MVPsSemana`), y se elige una foto del partido.

1. Busca en Google Imágenes, una búsqueda por MVP. Filtra por reciente si
   salen fotos viejas.
2. Elige dos o tres candidatas por jugador. Antes de proponerlas abre la página
   de origen y comprueba lo que se pueda: que es ese jugador (dorsal, nombre
   en el pie de foto), ese partido (rival y fecha) y que tiene buena resolución.
   Lo que no hayas podido confirmar, dilo.
3. Enséñaselas a Luis con captura y de dónde sale cada una. **Elige él.**
4. Con su sí para cada fichero (nombre, origen y tamaño), descárgala a
   `salidas/{año}/wNN/liga/mvp_fotos/` con un nombre claro
   (`mvp_ataque_mcmillan.jpg`). Solo imágenes (jpg, png, webp), nada más.
5. Súbela con `file_upload` en el tuit de ese MVP.

Si no aparece una foto clara de ese partido, no se pone una de otro: se le
dice a Luis y ese tuit lo decide él.

## 3. Elegir [A] o [B] y pedir el visto bueno

Enseña la lista numerada con las dos opciones de cada post y deja que Luis
elija ("todos los A", "A salvo el 3 y el 7", "sáltate el 5"). Después repite
en corto lo que va a salir y espera el sí. Los hilos largos se publican
seguidos; si Luis quiere parar en medio, se para y se retoma con `publicado`.

## 4. Navegador

Carga las herramientas en UNA llamada a ToolSearch (tabs_context_mcp,
tabs_create_mcp, navigate, computer, find, read_page, file_upload,
tabs_close_mcp). `tabs_context_mcp` primero; pestaña nueva; `x.com/home`.
Comprueba que la sesión abierta es **@CuartayDato**. Si es otra o no hay
sesión, para.

## 5. Post suelto (o el primero de un hilo)

1. Abre `https://x.com/compose/post`.
2. Clic en el cuadro de texto y escribe el texto del post.
3. Imágenes: localiza el `input` de fichero del compositor (`find`: "file input
   de imágenes del compositor") y súbelas con `file_upload`, todas en una
   llamada y en el orden del JSON (primero el resumen, luego la ficha). No
   hagas clic en el botón de la imagen: abre un diálogo del sistema que no ves.
4. **Comprueba antes de publicar**, con captura:
   - el texto es idéntico al de la cola, con sus saltos de párrafo;
   - las menciones y hashtags están escritos enteros y el desplegable de
     sugerencias está cerrado (si quedó abierto, no elijas ninguna sugerencia:
     ciérralo sin tocar el texto y vuelve a comprobar);
   - están todas las imágenes, en su orden, ya cargadas;
   - el contador no está en rojo.
5. Clic en "Postear".
6. Comprueba que salió y coge su URL (perfil `x.com/CuartayDato`, primer post).
7. Apúntalo:
   `python publicar_cola.py --marcar "<titulo exacto>" --letra A --url <url>`

## 6. Respuesta dentro de un hilo

1. Abre la URL del tuit anterior del hilo (la que acabas de apuntar).
2. Clic en responder. Se abre el compositor con una línea "Respondiendo a…".
3. **Desmarcar a los demás.** Si el tuit anterior mencionaba cuentas, la línea
   dice "Respondiendo a @CuartayDato y N más". Haz clic en esa línea: se abre
   el diálogo "Respondiendo a", con @CuartayDato arriba (fijo) y debajo
   "Otros en esta conversación" con una casilla general y una por cuenta.
   Desmarca la casilla general de "Otros en esta conversación" (se desmarcan
   todas las de debajo), comprueba con una captura que no queda NINGUNA
   marcada salvo la de @CuartayDato, y pulsa "Listo".
4. De vuelta en el compositor, la línea tiene que decir solo
   "Respondiendo a @CuartayDato". Compruébalo con captura (y `zoom` si hace falta).
   - Si el anterior no mencionaba a nadie (la presentación), la línea ya sale
     así: se comprueba igual.
   - Si no consigues dejarla así o no puedes leerla con seguridad: NO
     publiques. Para y cuéntaselo a Luis con la captura.
5. Escribe el texto y sube las imágenes como en el punto 5.
6. **Vuelve a comprobar la línea "Respondiendo a @CuartayDato"** justo antes de
   pulsar "Responder": al escribir o subir imágenes no debería cambiar, pero es
   la única regla que no admite fallo.
7. Publica, comprueba, coge la URL de ESTA respuesta y apúntala. La siguiente
   responde a esta.

Las menciones que van escritas en el texto del propio post (la afición del
equipo) sí salen: lo que se quita son las heredadas del post anterior.

## 7. Al terminar

Resume qué salió (con enlaces), qué se saltó y por qué, y qué queda pendiente.
Cierra la pestaña que abriste. Si paraste a mitad, di exactamente cuál fue el
último tuit publicado.

## Notas de campo

Lo que se vaya aprendiendo de la página de X (dónde está cada cosa, qué falla)
se apunta aquí, con fecha, para no redescubrirlo.

- 12-oct-2026: skill recién escrita, **todavía sin probar con un post real**.
  La primera vez, ir despacio y con captura en cada paso; confirmar sobre todo
  que `file_upload` acepta las rutas de `salidas/` y cómo se comportan los
  saltos de línea y el desplegable de menciones al escribir.
