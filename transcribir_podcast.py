# transcribir_podcast.py
# Transcribe un episodio de podcast con faster-whisper (14-oct-2026).
#
# Sustituye a la receta de `whisper` + ffmpeg en trozos: con int8 usa más o
# menos un tercio de la memoria y va 3-4 veces más rápido en CPU, así que el
# episodio entero cabe de una vez. Escribe cada frase al momento con su
# minuto: si el proceso muere a mitad, lo transcrito se queda en el fichero y
# `--desde` sigue desde el último minuto escrito.
#
#   python transcribir_podcast.py ep.mp3 salida.txt
#   python transcribir_podcast.py ep.mp3 salida.txt --desde 23:40
#   python transcribir_podcast.py ep.mp3 salida.txt --modelo small
#
# Modelo `base` por defecto (Luis, 22-sep-2026: los nombres se verifican por
# web de todas formas). El glosario NFL va en --glosario.

import argparse
import io
import sys
import time

GLOSARIO = ("Podcast de NFL en español. Quarterback, touchdown, field goal, "
            "red zone, tercer down, cuarto down, sack, interception, fumble, "
            "running back, wide receiver, tight end, linebacker, cornerback, "
            "safety, EPA, playoffs, Super Bowl.")


def a_segundos(txt):
    partes = [float(p) for p in txt.split(":")]
    seg = 0.0
    for p in partes:
        seg = seg * 60 + p
    return seg


def leer_audio(ruta):
    """Audio a 16 kHz mono float32 con ffmpeg. No se usa el decodificador de
    faster-whisper (PyAV): la version que instala pip el 14-oct-2026 no casa
    con faster-whisper 1.2.1 (open() got an unexpected keyword argument
    'metadata_errors')."""
    import subprocess
    import numpy as np
    crudo = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", ruta,
         "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(crudo, np.int16).astype(np.float32) / 32768.0


def minuto(seg):
    seg = int(seg)
    return f"{seg // 3600:d}:{seg % 3600 // 60:02d}:{seg % 60:02d}"


def main():
    ap = argparse.ArgumentParser(description="Transcribir un podcast con faster-whisper")
    ap.add_argument("audio")
    ap.add_argument("salida")
    ap.add_argument("--modelo", default="base")
    ap.add_argument("--glosario", default="",
                    help="nombres propios del episodio, se añaden al glosario NFL")
    ap.add_argument("--desde", default=None, help="mm:ss o h:mm:ss para continuar")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    modelo = WhisperModel(args.modelo, device="cpu", compute_type="int8", cpu_threads=4)
    prompt = (GLOSARIO + " " + args.glosario).strip()
    inicio = a_segundos(args.desde) if args.desde else 0.0
    clip = [inicio] if inicio else "0"

    t0 = time.time()
    segmentos, info = modelo.transcribe(
        leer_audio(args.audio), language="es", initial_prompt=prompt, vad_filter=True,
        beam_size=1, clip_timestamps=clip)
    modo = "a" if inicio else "w"
    ultimo = inicio
    with io.open(args.salida, modo, encoding="utf-8") as f:
        for s in segmentos:
            f.write(f"[{minuto(s.start)}] {s.text.strip()}\n")
            f.flush()
            ultimo = s.end
    dur = info.duration
    print(f"Hecho: {minuto(ultimo)} de {minuto(dur)} en {minuto(time.time() - t0)}"
          f" -> {args.salida}")
    if dur and ultimo < dur - 120:
        print("OJO: la transcripción acaba más de 2 min antes que el audio. "
              "Comprobar el final.", file=sys.stderr)


if __name__ == "__main__":
    main()
