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
# martes  -> dato de la semana, power rankings, MVPs (TXT), contenders (desde
#            la semana 5), bot: balance de la semana jugada + picks de la
#            proxima (TXT). Para los posts de
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
# Primera jornada jugada con la que el batch del martes saca contenders_tracker
CONTENDERS_DESDE = 5


def log(msg):
    linea = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linea, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with io.open(LOG, "a", encoding="utf-8") as f:
        f.write(linea + "\n")


def paso(nombre, args, stdin_text=None, captura=None, timeout=1200,
         sin_muestra=None):
    """Ejecuta un script del proyecto. Si `captura`, vuelca stdout a ese TXT.

    sin_muestra: codigo de salida que significa "aun no hay datos para esto"
    (el bot en las primeras jornadas). Se registra aparte y no cuenta como fallo.
    """
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
        if sin_muestra is not None and r.returncode == sin_muestra:
            cola = [l.strip() for l in (r.stdout or "").splitlines()
                    if l.strip()][-2:]
            log("   SIN MUESTRA (no es fallo): " + " | ".join(cola))
            return True
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
    # Cada dia en su fichero: cuando compartian borradores_posts.md, el batch
    # del martes borraba los posts de partido del lunes antes de publicarlos
    destino = os.path.join(txt_dir, "borradores_lunes.md"
                           if "lunes" in prompt_file else "borradores_posts.md")
    # No se borra: se aparta a _previo.md. Ahora que un batch muerto se
    # relanza (21-sep-2026), lo que hubiera ahi puede ser trabajo verificado
    # a mano, y el relanzamiento no puede llevarselo por delante
    if os.path.exists(destino):
        previo = destino.replace(".md", "_previo.md")
        if os.path.exists(previo):
            os.remove(previo)
        os.replace(destino, previo)
        log(f"   el borrador anterior se aparta en {os.path.basename(previo)}")
    log("-> borradores: claude -p (Read/Glob/Grep/Write/WebSearch)")
    try:
        # El prompt va por STDIN, nunca como argumento: en Windows `claude` es
        # claude.CMD y cmd.exe corta el argumento en el primer salto de linea.
        # Asi fallaron el 08-sep y el 15-sep ("esto es una ORDEN DE...").
        r = subprocess.run(
            [exe, "-p",
             "--allowedTools", "Read", "Glob", "Grep", "Write", "WebSearch"],
            cwd=RAIZ, input=prompt, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=3600)
        with io.open(destino.replace(".md", "_stdout.log"), "w",
                     encoding="utf-8") as f:
            f.write(r.stdout or "")
            if r.stderr:
                f.write("\n--- STDERR ---\n" + r.stderr)
        if r.returncode == 0 and os.path.exists(destino):
            log(f"   OK -> {destino}")
            return True
        cola = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        log(f"   FALLO (exit {r.returncode}): " + " | ".join(cola))
        return False
    except subprocess.TimeoutExpired:
        log("   FALLO: timeout de 3600s")
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


# Hora de cada tarea programada: (dia de la semana, hora, minuto). El batch
# "domingo" se lanza el SABADO por la noche.
HORARIO = {"lunes": (0, 10, 0), "martes": (1, 8, 0), "domingo": (5, 23, 0)}


INTENTOS_MAX = 2   # el original y UN relanzamiento: ni bucle ni gasto doble


def _lineas_desde(hora, marca):
    """Cuantas lineas del log con esa marca hay desde `hora`."""
    if not os.path.exists(LOG):
        return 0
    n = 0
    for linea in io.open(LOG, encoding="utf-8", errors="replace"):
        if marca in linea:
            try:
                ts = datetime.strptime(linea[1:20], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if ts >= hora:
                n += 1
    return n


def batch_pendiente(ahora=None):
    """El batch programado mas reciente, si no llego a TERMINAR; si no, None.

    Las tareas son de tipo Interactive (sin admin no se pueden cambiar): si a
    su hora no hay sesion iniciada, Windows las salta y StartWhenAvailable no
    las recupera. Paso el 15-sep-2026, tras un reinicio de Windows Update.
    Solo se recupera el ULTIMO: el martes regenera lo del lunes, y relanzar uno
    viejo pisaria borradores_posts.md de uno posterior.

    Desde el 21-sep-2026 lo que se busca es la linea de FIN, no la de inicio:
    ese dia el batch arranco, murio dentro del paso de borradores y se quedo
    sin escribir ni FALLO ni FIN, asi que contaba como hecho y nadie se entero.
    Un batch que empezo y no termino tambien es un batch pendiente. Para que
    eso no se convierta en un bucle, como mucho se relanza una vez
    (INTENTOS_MAX): si el segundo intento tampoco cierra, el log lo dice y lo
    miras tu.
    """
    from datetime import timedelta
    ahora = ahora or datetime.now()
    ultimos = {}
    for dia, (wd, h, m) in HORARIO.items():
        t = ahora.replace(hour=h, minute=m, second=0, microsecond=0)
        t -= timedelta(days=(ahora.weekday() - wd) % 7)
        if t > ahora:
            t -= timedelta(days=7)
        ultimos[dia] = t
    dia = max(ultimos, key=ultimos.get)
    hora = ultimos[dia]

    if _lineas_desde(hora, f"===== FIN BATCH {dia.upper()}"):
        return None                                   # cerrado: nada que hacer
    if _lineas_desde(hora, f"===== BATCH {dia.upper()} ") >= INTENTOS_MAX:
        log(f"RECUPERACION: el batch {dia} ya se intento {INTENTOS_MAX} veces "
            f"sin cerrar — no se relanza, revisalo a mano")
        return None
    return dia




def pasos_del_dia(dia, SEASON, W, txt_dir, ok):
    """Los pasos de ese dia, apilando su resultado en `ok` (que es del
    llamante a proposito: si esto revienta a mitad, main sigue sabiendo
    cuantos pasos habian salido bien y puede cerrar el log)."""
    if dia == "lunes":
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

    elif dia == "martes":
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
        # Contenders (formula del campeon): desde la semana 5. Antes, los 12
        # umbrales prorrateados a 17 partidos son ruido (decidido 24-sep-2026)
        if W >= CONTENDERS_DESDE:
            ok.append(paso("contenders (formula del campeon)",
                           ["contenders_tracker.py", "--season", str(SEASON),
                            "--week", str(W)],
                           captura=os.path.join(txt_dir, "contenders.txt")))
        else:
            log(f"-> contenders: se salta hasta la semana {CONTENDERS_DESDE} "
                f"(jugadas {W})")
        # JUEVES: bot — balance de la jugada y picks de la proxima, a TXT
        ok.append(paso("bot: balance semana jugada",
                       ["Manning_bot.py", "--no-retrain", "--week", str(W)],
                       captura=os.path.join(txt_dir, "bot_balance.txt"),
                       sin_muestra=3))
        ok.append(paso("bot: picks proxima jornada",
                       ["Manning_bot.py", "--no-retrain"],
                       captura=os.path.join(txt_dir, "bot_picks.txt"),
                       sin_muestra=3))
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
        # La semana NO es W+1: con el TNF del jueves ya jugado, W es la jornada
        # EN CURSO y salian las previas de la siguiente (paso el 19-sep-2026,
        # previas de la 3 la noche antes de la 2). Se pregunta al calendario
        # cual es la primera jornada con partidos por jugar.
        from pbp_loader import proxima_semana
        WP = proxima_semana() or (W + 1)
        log(f"previas: jornada por jugar = {WP} (ultima jugada: {W})")
        ok.append(paso("previas de la jornada",
                       ["Previas.py", "--week", str(WP)],
                       stdin_text="j\n\n\n", timeout=2400))


def main():
    ap = argparse.ArgumentParser(description="Batch semanal de generacion de PNGs")
    ap.add_argument("--dia", choices=["lunes", "martes", "domingo"])
    ap.add_argument("--recuperar", action="store_true",
                    help="al iniciar sesion: lanza el ultimo batch si se salto")
    args = ap.parse_args()
    if args.recuperar:
        args.dia = batch_pendiente()
        if not args.dia:
            return
        log(f"RECUPERACION: el batch {args.dia} no llego a arrancar a su hora")
    elif not args.dia:
        ap.error("hace falta --dia o --recuperar")

    # Semana jugada segun schedules (la fuente de verdad del proyecto)
    from pbp_loader import ultima_semana, temporada_actual
    W = ultima_semana()
    SEASON = temporada_actual()
    if not W:
        log(f"Sin jornadas jugadas en {SEASON} todavia — nada que generar.")
        return
    # Los TXT y los borradores viven en el cajon `textos/` de la semana
    # (21-sep-2026); los PNG estan en previas/, partidos/ y liga/
    txt_dir = os.path.join(RAIZ, "salidas", str(SEASON), f"w{W:02d}", "textos")
    os.makedirs(txt_dir, exist_ok=True)

    log(f"===== BATCH {args.dia.upper()} — NFL {SEASON}, semana jugada {W} =====")
    ok = []
    # El FIN se escribe SIEMPRE, tambien si un paso lanza: el 21-sep-2026 el
    # batch se fue dentro del paso de borradores sin dejar ni FALLO ni FIN, y
    # la recuperacion lo dio por hecho. Un kill duro sigue sin dejar rastro,
    # pero de eso ya se encarga batch_pendiente: sin FIN, pendiente.
    try:
        pasos_del_dia(args.dia, SEASON, W, txt_dir, ok)
    finally:
        buenos = sum(1 for x in ok if x)
        log(f"===== FIN BATCH {args.dia.upper()}: {buenos}/{len(ok)} pasos OK =====")
    if buenos < len(ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
