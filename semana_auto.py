# semana_auto.py — Nivel 1 de la automatizacion del calendario de posts.
# La maquina GENERA; los posts los revisa y publica Luis (regla del proyecto:
# verificacion triple antes de publicar — esto no publica nada).
#
#   python semana_auto.py --dia martes    # tras la jornada (datos completos)
#   python semana_auto.py --dia domingo   # sabado noche: previas de la jornada
#
# martes  -> dato de la semana, power rankings, MVPs (TXT), bot: balance de la
#            semana jugada + picks de la proxima (TXT). Para los posts de
#            martes (dato+resumen), miercoles (PR+MVPs) y jueves (bot).
# domingo -> previas de TODOS los partidos de la proxima jornada (PNGs + PDF)
#            para el hilo del domingo por la manana.
#
# El resumen del mejor partido NO esta aqui: elegir el partido es editorial.
# Cada paso es independiente: si uno falla, los demas siguen y el log lo dice.

import argparse
import io
import os
import subprocess
import sys
from datetime import datetime

RAIZ = os.path.dirname(os.path.abspath(__file__))
LOG  = os.path.join(RAIZ, "salidas", "auto_log.txt")


def log(msg):
    linea = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linea, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with io.open(LOG, "a", encoding="utf-8") as f:
        f.write(linea + "\n")


def paso(nombre, args, stdin_text=None, captura=None, timeout=1200):
    """Ejecuta un script del proyecto. Si `captura`, vuelca stdout a ese TXT."""
    log(f"-> {nombre}: {' '.join(args)}")
    try:
        r = subprocess.run([sys.executable] + args, cwd=RAIZ,
                           input=stdin_text, capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
        if captura:
            os.makedirs(os.path.dirname(captura), exist_ok=True)
            with io.open(captura, "w", encoding="utf-8") as f:
                f.write(r.stdout)
            log(f"   salida -> {captura}")
        if r.returncode != 0:
            cola = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
            log(f"   FALLO (exit {r.returncode}): " + " | ".join(cola))
            return False
        log(f"   OK")
        return True
    except subprocess.TimeoutExpired:
        log(f"   FALLO: timeout de {timeout}s")
        return False
    except Exception as e:
        log(f"   FALLO: {type(e).__name__}: {e}")
        return False


def borradores(txt_dir, W):
    """Redaccion de borradores con `claude -p` (sin publicar nada)."""
    import shutil
    from datetime import date
    exe = shutil.which("claude")
    if not exe:
        log("-> borradores: claude CLI no encontrado — paso omitido")
        return False
    plantilla = io.open(os.path.join(RAIZ, "borradores_prompt.md"),
                        encoding="utf-8").read()
    prompt = (plantilla.replace("{DIR}", txt_dir.replace(os.sep, "/"))
                       .replace("{W}", str(W))
                       .replace("{FECHA}", str(date.today())))
    # Si queda un borrador de una corrida anterior de esta misma semana, fuera:
    # su mera existencia contaria como exito aunque claude no escribiera nada
    destino = os.path.join(txt_dir, "borradores_posts.md")
    if os.path.exists(destino):
        os.remove(destino)
    log("-> borradores: claude -p (Read/Glob/Grep/Write/WebSearch)")
    try:
        r = subprocess.run(
            [exe, "-p", prompt,
             "--allowedTools", "Read", "Glob", "Grep", "Write", "WebSearch"],
            cwd=RAIZ, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=1800)
        if r.returncode == 0 and os.path.exists(destino):
            log(f"   OK -> {destino}")
            return True
        cola = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        log(f"   FALLO (exit {r.returncode}): " + " | ".join(cola))
        return False
    except subprocess.TimeoutExpired:
        log("   FALLO: timeout de 1800s")
        return False
    except Exception as e:
        log(f"   FALLO: {type(e).__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Batch semanal de generacion de PNGs")
    ap.add_argument("--dia", choices=["martes", "domingo"], required=True)
    args = ap.parse_args()

    # Semana jugada segun schedules (la fuente de verdad del proyecto)
    from pbp_loader import ultima_semana, temporada_actual
    W = ultima_semana()
    SEASON = temporada_actual()
    if not W:
        log(f"Sin jornadas jugadas en {SEASON} todavia — nada que generar.")
        return
    txt_dir = os.path.join(RAIZ, "salidas", str(SEASON), f"w{W:02d}")

    log(f"===== BATCH {args.dia.upper()} — NFL {SEASON}, semana jugada {W} =====")
    ok = []

    if args.dia == "martes":
        # Semaforo de fuentes primero: si algo esta caido, que quede en el log
        ok.append(paso("estado de datos", ["estado_datos.py"],
                       captura=os.path.join(txt_dir, "estado_datos.txt")))
        # MARTES: dato de la semana (el resumen del partido lo eliges tu)
        ok.append(paso("dato de la semana", ["DatoSemana.py", "--week", str(W)],
                       captura=os.path.join(txt_dir, "dato_semana.txt")))
        # MIERCOLES: power rankings + MVPs de la jornada (TXT para el post)
        ok.append(paso("power rankings", ["power_rankings.py", "--week", str(W)],
                       captura=os.path.join(txt_dir, "power_rankings.txt")))
        ok.append(paso("MVPs de la jornada", ["MVPsSemana.py", "--week", str(W)],
                       stdin_text="s\n",
                       captura=os.path.join(txt_dir, "mvps_semana.txt")))
        # JUEVES: bot — balance de la jugada y picks de la proxima, a TXT
        ok.append(paso("bot: balance semana jugada",
                       ["Manning_bot.py", "--no-retrain", "--week", str(W)],
                       captura=os.path.join(txt_dir, "bot_balance.txt")))
        ok.append(paso("bot: picks proxima jornada",
                       ["Manning_bot.py", "--no-retrain"],
                       captura=os.path.join(txt_dir, "bot_picks.txt")))
        # NIVEL 2: Claude Code headless redacta los borradores a partir de lo
        # generado. Solo puede leer, buscar en web y escribir; la publicacion
        # sigue siendo de Luis (verificacion triple del CLAUDE.md)
        ok.append(borradores(txt_dir, W))

    else:  # domingo (se lanza el sabado por la noche)
        # Previas de TODA la proxima jornada para el hilo del domingo.
        # Los PNGs se archivan en la semana W+1, que es la que previsualizan.
        ok.append(paso("previas de la jornada",
                       ["Previas.py", "--week", str(W + 1)],
                       stdin_text="j\n\n\n", timeout=2400))

    buenos = sum(1 for x in ok if x)
    log(f"===== FIN: {buenos}/{len(ok)} pasos OK =====")
    if buenos < len(ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
