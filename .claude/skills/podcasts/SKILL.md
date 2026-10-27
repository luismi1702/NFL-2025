---
name: podcasts
description: Podcasts y programas de YouTube de NFL que sigue @CuartayDato (Ecos de la NFL, 100 Yardas, El Capologist/Previologist, El Nickel). Úsala cuando Luis pregunte "¿hay podcasts nuevos?", "¿qué ha salido?", "/podcasts", pase el enlace de un episodio (PepeDiario u otro) o quiera las respuestas a un programa. Explica el automático y cómo consultarlo o forzarlo.
---

# /podcasts — los programas se escuchan solos

Desde el 19-oct-2026 (Luis: "no puede ser que tenga que estar pendiente de
todo") los podcasts NO se miran a mano. La tarea de Windows
**"NFL2025 podcasts"** lanza `python podcasts_auto.py` cada hora de 9:05 a
23:05. Con cada episodio nuevo:

1. Lo detecta en su feed (`FUENTES` en `podcasts_auto.py`).
2. Lo transcribe: YouTube con subtítulos automáticos (`yt-dlp`) y, si no hay,
   audio con `transcribir_podcast.py` (faster-whisper `base`); RSS con el mp3.
   Queda en `salidas/podcasts/AAAAMMDD_programa_titulo.txt`.
3. Un `claude -p` con `podcasts_prompt.md` saca el ángulo, lo mide con el PBP
   (scripts en `lab/podcasts/`), escribe la respuesta [A]/[B] en
   `salidas/{año}/wNN/textos/borradores_podcast_<programa>_<MMDD>.md` y añade
   la fila a `docs/podcasts_revisados.md` (también si no hay pieza).
4. Pasa el verificador de siempre (`semana_auto.verificar`) y se regenera la
   cola. La tarjeta sale como MANUAL ("Respuesta a …"): la publica Luis
   respondiendo al tuit del episodio. `/publicar` no la toca.

Lo que se descubre con más de 4 días (`CADUCA_DIAS`) se marca "caducado" y
no se procesa: una previa de una jornada ya jugada llega tarde. En El Nickel
se saltan los shorts (título con #) y lo que dura menos de 15 min.

## Comandos

```
python podcasts_auto.py --estado        # últimos 25: qué salió, qué falló
python podcasts_auto.py --solo-mirar    # qué hay nuevo, sin procesar
python podcasts_auto.py                 # procesar lo nuevo ahora
python podcasts_auto.py --forzar <id|videoId|url>   # uno concreto, aunque esté
                                                    # visto o caducado
```

Un enlace suelto (PepeDiario, un vídeo fuera de los feeds) también va con
`--forzar <url>`. Tarda: transcribir 2-6 min, redactar y verificar 10-20 min.
Lánzalo en segundo plano y avisa al terminar.

## Cuando Luis pregunte "¿hay podcasts?"

1. `python podcasts_auto.py --estado` y `--solo-mirar`.
2. Lo procesado con pieza: enséñale las opciones [A]/[B] del borrador, con el
   enlace del episodio para que responda (regla del 05-oct: siempre las dos
   opciones redactadas, sin hashtags ni menciones).
3. "sin pieza": dile en una línea por qué (lo pone la fila de
   `docs/podcasts_revisados.md`).
4. "error": mira `salidas/auto_log.txt` (líneas `[podcasts]`) y el
   `_redactor.log` junto a la transcripción; arréglalo y `--forzar`.

## Añadir o quitar un programa

Se edita `FUENTES` en `podcasts_auto.py` (RSS o canal de YouTube; los RSS se
sacan con la API de búsqueda de iTunes) y se siembra lo viejo para que no
procese el archivo entero: `python podcasts_auto.py --sembrar` (solo marca lo
que aún no está en el estado). Apuntarlo en la memoria
`project_reaccion_podcasts`.
