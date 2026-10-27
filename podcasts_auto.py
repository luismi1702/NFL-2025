# podcasts_auto.py
# Podcasts sin estar pendiente (Luis, 19-oct-2026: "no puede ser que tenga que
# estar pendiente de todo"). Mira los feeds de los programas que seguimos, y
# cada episodio NUEVO lo transcribe, lo pasa a un `claude -p` que saca su
# angulo y lo mide con nuestro PBP, y deja la respuesta [A]/[B] en la cola de
# la semana (tarjeta manual: la publica Luis respondiendo al tuit del
# episodio). Despues pasa el verificador de siempre (semana_auto.verificar).
#
# Lo lanza la tarea "NFL2025 podcasts" cada hora de 9:00 a 23:00: mirar un
# feed cuesta nada, y solo se trabaja cuando hay algo nuevo.
#
#   python podcasts_auto.py              # mirar feeds y procesar lo nuevo
#   python podcasts_auto.py --estado     # que hay, que se hizo, que falta
#   python podcasts_auto.py --solo-mirar # detectar sin procesar
#   python podcasts_auto.py --forzar <id o url>   # procesar uno aunque este
#                                                 # visto o caducado
#
# Estado en salidas/podcasts/estado.json (id del episodio -> que paso) y
# transcripciones en salidas/podcasts/. El registro editorial sigue siendo
# docs/podcasts_revisados.md: lo rellena el redactor.

import argparse
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

RAIZ = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(RAIZ, "salidas", "podcasts")
ESTADO = os.path.join(DIR, "estado.json")
LOCK = os.path.join(DIR, "podcasts.lock")
UA = {"User-Agent": "Mozilla/5.0"}

# Que se sigue (de project_reaccion_podcasts y docs/podcasts_revisados.md).
# `incluir`: regex sobre el titulo; lo demas del feed no se mira.
# PepeDiario no tiene feed: Luis pasa el enlace (--forzar <url>).
FUENTES = [
    {"programa": "Ecos de la NFL", "tipo": "rss",
     "url": "https://api.substack.com/feed/podcast/5284175/s/273417.rss"},
    {"programa": "100 Yardas", "tipo": "rss",
     "url": "https://fapi-top.prisasd.com/podcast/playser/100_yardas.xml"},
    # Del feed del Capologist solo el programa del lunes y el del viernes
    {"programa": "El Capologist", "tipo": "rss",
     "url": "https://feeds.ivoox.com/feed_fg_f1370648_filtro_1.xml",
     "incluir": r"^El (Capologist|Previologist) \d"},
    # El Nickel: el lunes y la previa. Fuera los shorts (llevan #) y lo corto
    {"programa": "El Nickel", "tipo": "youtube",
     "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCTD2RyzRb439yP6MvT5WTJQ",
     "excluir": r"#", "min_minutos": 15},
]

# Un episodio que se descubre con mas de 4 dias ya no se responde: una previa
# de una jornada jugada o un lunes de hace una semana llegan tarde
CADUCA_DIAS = 4
DIAS = ["LUNES", "MARTES", "MIÉRCOLES", "JUEVES", "VIERNES", "SÁBADO", "DOMINGO"]
TOOLS = ["Read", "Glob", "Grep", "Write", "Edit", "WebSearch",
         "Bash(python:*)", "Bash(mkdir:*)"]


def log(msg):
    sys.path.insert(0, RAIZ)
    from semana_auto import log as _log
    _log("[podcasts] " + msg)


def leer_estado():
    if os.path.exists(ESTADO):
        return json.load(io.open(ESTADO, encoding="utf-8"))
    return {}


def guardar_estado(est):
    os.makedirs(DIR, exist_ok=True)
    tmp = ESTADO + ".tmp"
    json.dump(est, io.open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, ESTADO)


def bajar(url, timeout=30):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                  timeout=timeout).read()


def episodios(f):
    """Lista de episodios del feed: id, titulo, fecha (UTC), url, audio."""
    raiz = ET.fromstring(bajar(f["url"]))
    out = []
    if f["tipo"] == "youtube":
        ns = {"a": "http://www.w3.org/2005/Atom",
              "yt": "http://www.youtube.com/xml/schemas/2015"}
        for e in raiz.findall("a:entry", ns):
            vid = e.find("yt:videoId", ns).text
            out.append({"id": "yt:" + vid, "titulo": e.find("a:title", ns).text,
                        "fecha": datetime.fromisoformat(e.find("a:published", ns).text),
                        "url": f"https://www.youtube.com/watch?v={vid}",
                        "audio": None})
    else:
        for it in raiz.iter("item"):
            enc = it.find("enclosure")
            guid = it.findtext("guid") or (enc.get("url") if enc is not None else it.findtext("title"))
            out.append({"id": "rss:" + guid.strip(), "titulo": (it.findtext("title") or "").strip(),
                        "fecha": parsedate_to_datetime(it.findtext("pubDate")),
                        "url": (it.findtext("link") or "").strip(),
                        "audio": enc.get("url") if enc is not None else None})
    inc, exc = f.get("incluir"), f.get("excluir")
    out = [e for e in out if (not inc or re.search(inc, e["titulo"]))
           and (not exc or not re.search(exc, e["titulo"]))]
    for e in out:
        e["programa"] = f["programa"]
        e["tipo"] = f["tipo"]
        e["min_minutos"] = f.get("min_minutos")
    return out


def slug(txt, n=40):
    s = re.sub(r"[^a-z0-9]+", "_", txt.lower().encode("ascii", "ignore").decode())
    return s.strip("_")[:n]


def vtt_a_texto(ruta):
    """Subtitulos automaticos de YouTube: cada linea sale tres veces y con
    etiquetas de tiempo dentro. Se deduplica LINEA a LINEA, con su minuto."""
    out, previa, minuto = [], None, "0:00:00"
    for linea in io.open(ruta, encoding="utf-8"):
        linea = linea.strip()
        m = re.match(r"(\d+):(\d\d):(\d\d)\.\d+ -->", linea)
        if m:
            minuto = f"{int(m.group(1))}:{m.group(2)}:{m.group(3)}"
            continue
        if not linea or linea.startswith(("WEBVTT", "Kind:", "Language:")) or "-->" in linea:
            continue
        limpia = re.sub(r"<[^>]+>", "", linea).strip()
        if limpia and limpia != previa:
            out.append(f"[{minuto}] {limpia}")
            previa = limpia
    return "\n".join(out) + "\n"


def transcribir(ep, base):
    """Deja la transcripcion en base + '.txt'. YouTube: subtitulos automaticos
    y, si no hay (o YouTube da 429), audio + faster-whisper. RSS: el mp3."""
    txt = base + ".txt"
    if os.path.exists(txt) and os.path.getsize(txt) > 2000:
        return txt
    audio = base + ".mp3"
    if ep["tipo"] == "youtube":
        subprocess.run(["yt-dlp", "--skip-download", "--write-auto-subs",
                        "--sub-langs", "es.*", "--sub-format", "vtt",
                        "-o", base + ".%(ext)s", ep["url"]],
                       capture_output=True, text=True, timeout=300)
        vtts = sorted(glob.glob(base + "*.vtt"))
        if vtts:
            io.open(txt, "w", encoding="utf-8").write(vtt_a_texto(vtts[0]))
            if os.path.getsize(txt) > 2000:
                log(f"   subtitulos de YouTube -> {os.path.basename(txt)}")
                return txt
        log("   sin subtitulos: audio + faster-whisper")
        subprocess.run(["yt-dlp", "-x", "--audio-format", "mp3",
                        "-o", base + ".%(ext)s", ep["url"]],
                       capture_output=True, text=True, timeout=1800)
    elif ep["audio"]:
        with io.open(audio, "wb") as f:
            f.write(bajar(ep["audio"], timeout=600))
    if not os.path.exists(audio):
        raise RuntimeError("no se pudo bajar el audio")
    r = subprocess.run([sys.executable, os.path.join(RAIZ, "transcribir_podcast.py"),
                        audio, txt], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=5400)
    if r.returncode != 0 or not os.path.exists(txt):
        raise RuntimeError("transcribir_podcast.py: " + (r.stderr or r.stdout)[-300:])
    log("   " + (r.stdout.strip().splitlines() or ["transcrito"])[-1])
    try:
        os.remove(audio)        # el mp3 pesa; la transcripcion es lo que vale
    except OSError:
        pass
    return txt


def duracion_minutos(url):
    r = subprocess.run(["yt-dlp", "--print", "duration", "--skip-download", url],
                       capture_output=True, text=True, timeout=120)
    try:
        return float(r.stdout.strip().splitlines()[-1]) / 60
    except (ValueError, IndexError):
        return None


def carpeta_textos():
    """Carpeta de la cola de HOY: la semana de la cola va de jueves a
    miercoles. Lunes-miercoles, la jornada jugada; jueves-domingo, la que se
    juega (como dato() en semana_auto)."""
    sys.path.insert(0, RAIZ)
    from pbp_loader import ultima_semana, proxima_semana, temporada_actual
    season = temporada_actual()
    if datetime.now().weekday() <= 2:
        w = ultima_semana()
    else:
        w = proxima_semana() or (ultima_semana() + 1)
    return season, w, os.path.join(RAIZ, "salidas", str(season), f"w{w:02d}", "textos")


def redactar(ep, txt):
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("claude CLI no encontrado")
    season, w, txt_dir = carpeta_textos()
    os.makedirs(txt_dir, exist_ok=True)
    fecha_ep = ep["fecha"].astimezone().strftime("%d-%b-%Y")
    destino = os.path.join(txt_dir, f"borradores_podcast_{slug(ep['programa'], 20)}_"
                                    f"{ep['fecha'].astimezone():%m%d}.md")
    hoy = datetime.now()
    prompt = io.open(os.path.join(RAIZ, "podcasts_prompt.md"), encoding="utf-8").read()
    for k, v in {"{PROGRAMA}": ep["programa"], "{EPISODIO}": ep["titulo"],
                 "{FECHA_EP}": fecha_ep, "{URL}": ep["url"],
                 "{TRANSCRIPCION}": os.path.relpath(txt, RAIZ).replace(os.sep, "/"),
                 "{FECHA}": f"{hoy:%Y-%m-%d}", "{DIA}": DIAS[hoy.weekday()],
                 "{FECHA_FILA}": f"{hoy:%d-%b-%Y}".lower().replace(".", ""),
                 "{DESTINO}": os.path.relpath(destino, RAIZ).replace(os.sep, "/")}.items():
        prompt = prompt.replace(k, v)
    log(f"   redactor: claude -p -> {os.path.basename(destino)}")
    # Prompt por STDIN (en Windows claude es un .CMD y corta los argumentos)
    r = subprocess.run([exe, "-p", "--allowedTools", *TOOLS], cwd=RAIZ,
                       input=prompt, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=3600)
    io.open(os.path.splitext(txt)[0] + "_redactor.log", "w", encoding="utf-8").write(
        (r.stdout or "") + ("\n--- STDERR ---\n" + r.stderr if r.stderr else ""))
    if r.returncode != 0:
        raise RuntimeError(f"redactor exit {r.returncode}")
    if not os.path.exists(destino):
        log("   sin pieza (nada pasa el liston); fila en podcasts_revisados.md")
        return "sin pieza", None
    sys.path.insert(0, RAIZ)
    from semana_auto import verificar
    verificar(destino)
    subprocess.run([sys.executable, os.path.join(RAIZ, "cola_posts.py"),
                    "--season", str(season), "--week", str(w)],
                   cwd=RAIZ, capture_output=True, timeout=600)
    texto = io.open(destino, encoding="utf-8").read()
    ok = "VERIFICADO:" in texto
    log(f"   pieza en la cola w{w:02d}" + ("" if ok else " (SIN VERIFICAR: bloqueada)"))
    return ("pieza" if ok else "pieza sin verificar"), os.path.relpath(destino, RAIZ)


def procesar(ep, est):
    base = os.path.join(DIR, f"{ep['fecha'].astimezone():%Y%m%d}_{slug(ep['programa'], 20)}_"
                             f"{slug(ep['titulo'], 30)}")
    os.makedirs(DIR, exist_ok=True)
    log(f"-> {ep['programa']}: {ep['titulo']}")
    reg = est.setdefault(ep["id"], {})
    reg.update({"programa": ep["programa"], "titulo": ep["titulo"], "url": ep["url"],
                "fecha": ep["fecha"].isoformat(), "estado": "en curso",
                "intentos": reg.get("intentos", 0) + 1,
                "visto": reg.get("visto") or datetime.now().isoformat(timespec="minutes")})
    guardar_estado(est)
    try:
        txt = transcribir(ep, base)
        reg["transcripcion"] = os.path.relpath(txt, RAIZ)
        reg["estado"], reg["borrador"] = redactar(ep, txt)
    except Exception as e:
        reg["estado"] = "error"
        reg["error"] = f"{type(e).__name__}: {e}"[:300]
        log(f"   FALLO: {reg['error']}")
    guardar_estado(est)


def main():
    ap = argparse.ArgumentParser(description="Podcasts automaticos")
    ap.add_argument("--estado", action="store_true")
    ap.add_argument("--solo-mirar", action="store_true")
    ap.add_argument("--forzar", default=None, help="id del estado, videoId o url")
    ap.add_argument("--sembrar", action="store_true",
                    help="marca como vistos todos los episodios actuales sin procesarlos")
    args = ap.parse_args()
    est = leer_estado()

    if args.estado:
        for k, v in sorted(est.items(), key=lambda kv: kv[1].get("fecha", ""), reverse=True)[:25]:
            print(f"{v.get('fecha', '')[:10]} | {v.get('programa', ''):15} | {v.get('estado', ''):20} | "
                  f"{v.get('titulo', '')[:60]}" + (f" | {v['borrador']}" if v.get("borrador") else "")
                  + (f" | {v['error']}" if v.get("error") else ""))
        return

    # Una sola corrida a la vez: un episodio de 2 h puede tardar mas de una
    # hora entre transcribir, redactar y verificar
    os.makedirs(DIR, exist_ok=True)
    if os.path.exists(LOCK):
        edad = datetime.now().timestamp() - os.path.getmtime(LOCK)
        if edad < 3 * 3600:
            print("Ya hay una corrida en marcha.")
            return
    io.open(LOCK, "w").write(str(os.getpid()))
    try:
        ahora = datetime.now(timezone.utc)
        todos = []
        for f in FUENTES:
            try:
                todos += episodios(f)
            except Exception as e:
                log(f"feed de {f['programa']} no responde: {type(e).__name__}: {e}")

        if args.sembrar:
            for ep in todos:
                est.setdefault(ep["id"], {"programa": ep["programa"], "titulo": ep["titulo"],
                                          "url": ep["url"], "fecha": ep["fecha"].isoformat(),
                                          "estado": "previo (sembrado)"})
            guardar_estado(est)
            print(f"Sembrados {len(todos)} episodios.")
            return

        if args.forzar:
            f = args.forzar
            sel = [e for e in todos if f in (e["id"], e["url"]) or e["id"].endswith(f)]
            if not sel and f.startswith("http"):
                # Enlace suelto (PepeDiario, o un YouTube fuera de los feeds)
                sel = [{"id": "url:" + f, "titulo": f, "programa": "Podcast",
                        "fecha": ahora, "url": f, "audio": None,
                        "tipo": "youtube", "min_minutos": None}]
            if not sel:
                print("No encuentro ese episodio en los feeds.")
                return
            procesar(sel[0], est)
            return

        nuevos = [e for e in todos if e["id"] not in est
                  or (est[e["id"]].get("estado") == "error" and est[e["id"]].get("intentos", 0) < 3)]
        if not nuevos:
            print("Nada nuevo.")
            return
        for ep in sorted(nuevos, key=lambda e: e["fecha"]):
            if ahora - ep["fecha"] > timedelta(days=CADUCA_DIAS) and ep["id"] not in est:
                est[ep["id"]] = {"programa": ep["programa"], "titulo": ep["titulo"],
                                 "url": ep["url"], "fecha": ep["fecha"].isoformat(),
                                 "estado": "caducado"}
                log(f"caducado (>{CADUCA_DIAS} dias): {ep['programa']} — {ep['titulo']}")
                continue
            if ep.get("min_minutos"):
                dur = duracion_minutos(ep["url"])
                if dur is not None and dur < ep["min_minutos"]:
                    est[ep["id"]] = {"programa": ep["programa"], "titulo": ep["titulo"],
                                     "url": ep["url"], "fecha": ep["fecha"].isoformat(),
                                     "estado": f"corto ({dur:.0f} min)"}
                    continue
            if args.solo_mirar:
                print(f"NUEVO: {ep['programa']} — {ep['titulo']} ({ep['fecha']:%d-%m %H:%M})")
                continue
            procesar(ep, est)
        guardar_estado(est)
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass


if __name__ == "__main__":
    main()
