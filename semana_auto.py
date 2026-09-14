# semana_auto.py — Nivel 1 de la automatizacion del calendario de posts.
# La maquina GENERA; los posts los revisa y publica Luis (regla del proyecto:
# verificacion triple antes de publicar — esto no publica nada).
#
#   python semana_auto.py --dia lunes     # manana siguiente a la jornada
#   python semana_auto.py --dia martes    # tras la jornada (datos completos)
#   python semana_auto.py --dia domingo   # sabado noche: previas de la jornada
#
# lunes   -> visuales de TODOS los partidos del domingo (resumenes + fichas
#            tacticas + bajo centro), destacados.txt con lo que merece post y
#            los borradores de esos posts. El Monday Night no esta: se juega esa
#            noche y entra en el batch del martes.
# martes  -> dato de la semana, power rankings, MVPs (TXT), bot: balance de la
#            semana jugada + picks de la proxima (TXT). Para los posts de
#            martes (dato+resumen), miercoles (PR+MVPs) y jueves (bot).
# domingo -> previas de TODOS los partidos de la proxima jornada (PNGs + PDF)
#            para el hilo del domingo por la manana.
#
# Los resumenes de partido SI estan aqui, y de TODOS los partidos: generarlos
# no es editorial, elegir cual se publica si. Salen numerados por orden de
# kickoff, asi que la carpeta de la semana se lee como se jugo la jornada.
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


def borradores(txt_dir, W, prompt_file="borradores_prompt.md"):
    """Redaccion de borradores con `claude -p` (sin publicar nada).

    prompt_file: el lunes usa `borradores_prompt_lunes.md` (posts de partido a
    partir de destacados.txt); el martes, el de siempre.
    """
    import shutil
    from datetime import date
    exe = shutil.which("claude")
    if not exe:
        log("-> borradores: claude CLI no encontrado — paso omitido")
        return False
    plantilla = io.open(os.path.join(RAIZ, prompt_file),
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


def resumenes(SEASON, W):
    """Los 2 PNGs de resumen de CADA partido de la jornada jugada.

    Un partido caido no tumba a los demas: se anota en el log y el paso termina
    diciendo cuantos salieron. Los nombres llevan delante el orden de kickoff
    (ver orden_partido en pbp_loader), asi que no hace falta ordenarlos luego.
    """
    from pbp_loader import cargar_pbp
    try:
        df, _ = cargar_pbp(SEASON, columns=["week", "game_id", "home_team",
                                            "away_team"],
                           solo_reg=False, avisar=False)
    except Exception as e:
        log(f"-> resumenes: no se pudo leer el PBP ({type(e).__name__}) — omitido")
        return False

    jornada = df[df["week"] == W].drop_duplicates("game_id").sort_values("game_id")
    if jornada.empty:
        log(f"-> resumenes: sin partidos con datos en la semana {W}")
        return False

    log(f"-> resumenes de partido: {len(jornada)} partidos de la semana {W}")
    fallos = []
    for fila in jornada.itertuples(index=False):
        vis, loc = fila.away_team, fila.home_team
        try:
            r = subprocess.run(
                [sys.executable, "resumen_partido.py",
                 "--season", str(SEASON), "--week", str(W)],
                cwd=RAIZ, input=f"{vis}\n{loc}\n", capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=600)
            hechos = [l for l in (r.stdout or "").splitlines()
                      if l.startswith("Guardado")]
            if r.returncode == 0 and len(hechos) == 2:
                log(f"   OK {vis}@{loc}")
                continue
            cola = (r.stderr or r.stdout or "").strip().splitlines()[-1:]
            log(f"   FALLO {vis}@{loc}: " + " | ".join(cola))
        except subprocess.TimeoutExpired:
            log(f"   FALLO {vis}@{loc}: timeout de 600s")
        except Exception as e:
            log(f"   FALLO {vis}@{loc}: {type(e).__name__}: {e}")
        fallos.append(f"{vis}@{loc}")

    log(f"   {len(jornada) - len(fallos)}/{len(jornada)} resumenes generados")
    return not fallos


def fichas(SEASON, W):
    """La ficha tactica de CADA partido de la jornada (cara a cara de facetas).

    Misma mecanica que resumenes(): un partido caido no tumba a los demas.
    """
    from pbp_loader import cargar_pbp
    try:
        df, _ = cargar_pbp(SEASON, columns=["week", "game_id", "home_team",
                                            "away_team"],
                           solo_reg=False, avisar=False)
    except Exception as e:
        log(f"-> fichas: no se pudo leer el PBP ({type(e).__name__}) — omitido")
        return False

    jornada = df[df["week"] == W].drop_duplicates("game_id").sort_values("game_id")
    if jornada.empty:
        log(f"-> fichas: sin partidos con datos en la semana {W}")
        return False

    log(f"-> fichas tacticas: {len(jornada)} partidos de la semana {W}")
    fallos = []
    for fila in jornada.itertuples(index=False):
        vis, loc = fila.away_team, fila.home_team
        try:
            r = subprocess.run(
                [sys.executable, "ficha_tactica.py",
                 "--season", str(SEASON), "--week", str(W)],
                cwd=RAIZ, input=f"{vis}\n{loc}\n", capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=600)
            if r.returncode == 0 and any(l.startswith("Guardado")
                                         for l in (r.stdout or "").splitlines()):
                log(f"   OK {vis}@{loc}")
                continue
            cola = (r.stderr or r.stdout or "").strip().splitlines()[-1:]
            log(f"   FALLO {vis}@{loc}: " + " | ".join(cola))
        except subprocess.TimeoutExpired:
            log(f"   FALLO {vis}@{loc}: timeout de 600s")
        except Exception as e:
            log(f"   FALLO {vis}@{loc}: {type(e).__name__}: {e}")
        fallos.append(f"{vis}@{loc}")

    log(f"   {len(jornada) - len(fallos)}/{len(jornada)} fichas generadas")
    return not fallos


def main():
    ap = argparse.ArgumentParser(description="Batch semanal de generacion de PNGs")
    ap.add_argument("--dia", choices=["lunes", "martes", "domingo"], required=True)
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

    if args.dia == "lunes":
        # LUNES: la jornada del domingo ya esta publicada. Visuales de todos los
        # partidos, rastreo de lo destacado y borradores de posts de partido.
        ok.append(paso("estado de datos", ["estado_datos.py"],
                       captura=os.path.join(txt_dir, "estado_datos.txt")))
        ok.append(resumenes(SEASON, W))
        ok.append(fichas(SEASON, W))
        ok.append(paso("bajo centro", ["under_center.py", "--season", str(SEASON),
                                       "--week", str(W)]))
        ok.append(paso("destacados de la jornada",
                       ["destacados.py", "--season", str(SEASON), "--week", str(W)],
                       captura=os.path.join(txt_dir, "destacados.txt")))
        hay_borradores = borradores(txt_dir, W, "borradores_prompt_lunes.md")
        ok.append(hay_borradores)
        if hay_borradores:
            ok.append(paso("cola de posts (copiar y pegar)",
                           ["cola_posts.py", "--season", str(SEASON),
                            "--week", str(W)]))

    elif args.dia == "martes":
        # Semaforo de fuentes primero: si algo esta caido, que quede en el log
        ok.append(paso("estado de datos", ["estado_datos.py"],
                       captura=os.path.join(txt_dir, "estado_datos.txt")))
        # MARTES: dato de la semana + los resumenes de TODA la jornada
        # (cual se publica lo eliges tu; generarlos todos no cuesta decision)
        ok.append(paso("dato de la semana", ["DatoSemana.py", "--week", str(W)],
                       captura=os.path.join(txt_dir, "dato_semana.txt")))
        # Se regeneran resumenes Y fichas: el lunes faltaba el Monday Night, que
        # a estas horas ya esta publicado. Los demas partidos salen identicos.
        ok.append(resumenes(SEASON, W))
        ok.append(fichas(SEASON, W))
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
        hay_borradores = borradores(txt_dir, W)
        ok.append(hay_borradores)
        # NIVEL 2.5: pagina de copiar y pegar a partir de esos borradores.
        # Sin borradores no hay nada que maquetar, asi que no cuenta como fallo.
        if hay_borradores:
            ok.append(paso("cola de posts (copiar y pegar)",
                           ["cola_posts.py", "--season", str(SEASON),
                            "--week", str(W)]))

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
