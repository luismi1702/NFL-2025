# publicar_cola.py
# -----------------------------------------------------------------------------
# Lo que la skill /publicar necesita de la cola: que toca publicar un dia y que
# ha salido ya. No publica nada (eso lo hace Claude en el navegador, con el
# visto bueno de Luis): lista y apunta.
#
#   python publicar_cola.py                 -> lo pendiente de HOY, en JSON
#   python publicar_cola.py --dia martes    -> lo de otro dia
#   python publicar_cola.py --week 5        -> forzar la carpeta de la semana
#   python publicar_cola.py --marcar "LUNES — Titulo" --letra A --url https://x.com/...
#
# La cola va de jueves a miercoles y vive en salidas/{año}/wNN/: se regenera
# la de las dos ultimas carpetas y se usa la mas nueva que tenga posts ese dia.
# Lo publicado se apunta en wNN/textos/publicados.json.
# -----------------------------------------------------------------------------
import argparse
import datetime
import io
import json
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

RAIZ = os.path.dirname(os.path.abspath(__file__))
DIAS = ["LUNES", "MARTES", "MIÉRCOLES", "JUEVES", "VIERNES", "SÁBADO", "DOMINGO"]


def norm(t):
    return (t.upper().replace("Á", "A").replace("É", "E").replace("Í", "I")
            .replace("Ó", "O").replace("Ú", "U").strip())


def semanas(season):
    base = os.path.join(RAIZ, "salidas", str(season))
    if not os.path.isdir(base):
        return []
    return sorted(int(m.group(1)) for m in (re.match(r"w(\d\d)$", d) for d in os.listdir(base)) if m)


def cargar(season, week):
    """Regenera la cola de esa semana y devuelve (carpeta, json)."""
    carpeta = os.path.join(RAIZ, "salidas", str(season), f"w{week:02d}")
    subprocess.run([sys.executable, os.path.join(RAIZ, "cola_posts.py"),
                    "--season", str(season), "--week", str(week)],
                   cwd=RAIZ, capture_output=True)
    ruta = os.path.join(carpeta, "textos", "cola_posts.json")
    if not os.path.exists(ruta):
        return carpeta, None
    return carpeta, json.load(io.open(ruta, encoding="utf-8"))


def publicados(carpeta):
    ruta = os.path.join(carpeta, "textos", "publicados.json")
    return json.load(io.open(ruta, encoding="utf-8")) if os.path.exists(ruta) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--dia", default=None)
    ap.add_argument("--marcar", default=None, help="titulo exacto de la seccion publicada")
    ap.add_argument("--letra", default="A")
    ap.add_argument("--url", default="")
    ap.add_argument("--descartar", action="store_true",
                    help="con --marcar: la seccion no se publica (Luis la descarta)")
    args = ap.parse_args()

    hoy = datetime.date.today()
    season = args.season or (hoy.year if hoy.month >= 3 else hoy.year - 1)
    dia = norm(args.dia) if args.dia else norm(DIAS[hoy.weekday()])
    dia = next((d for d in DIAS if norm(d).startswith(dia[:3])), None)
    if not dia:
        print(json.dumps({"error": f"dia no reconocido: {args.dia}"}, ensure_ascii=False))
        return 1

    candidatas = [args.week] if args.week else semanas(season)[-2:][::-1]
    elegido = None
    for w in candidatas:
        carpeta, datos = cargar(season, w)
        if datos and any(norm(c["dia"]) == norm(dia) for c in datos["cola"]):
            elegido = (w, carpeta, datos)
            break
    if not elegido:
        print(json.dumps({"dia": dia, "semanas_miradas": candidatas, "posts": [],
                          "aviso": "no hay borradores para ese dia"}, ensure_ascii=False))
        return 0
    w, carpeta, datos = elegido
    hechos = publicados(carpeta)

    if args.marcar:
        if not any(c["titulo"] == args.marcar for c in datos["cola"]):
            print(json.dumps({"error": "ese titulo no esta en la cola",
                              "titulo": args.marcar}, ensure_ascii=False))
            return 1
        hechos = [h for h in hechos if h["titulo"] != args.marcar]
        if args.descartar:
            hechos.append({"titulo": args.marcar, "descartado": True, "url": "",
                           "cuando": datetime.datetime.now().isoformat(timespec="seconds")})
        else:
            hechos.append({"titulo": args.marcar, "letra": args.letra.upper(), "url": args.url,
                           "cuando": datetime.datetime.now().isoformat(timespec="seconds")})
        with io.open(os.path.join(carpeta, "textos", "publicados.json"), "w",
                     encoding="utf-8", newline="\n") as f:
            json.dump(hechos, f, ensure_ascii=False, indent=1)
        print(json.dumps({"ok": True, "publicados": len(hechos)}, ensure_ascii=False))
        return 0

    ya = {h["titulo"]: h for h in hechos}
    posts = []
    for n, c in enumerate([c for c in datos["cola"] if norm(c["dia"]) == norm(dia)], 1):
        c = dict(c, orden=n, hilo="hilo" in c["titulo"].lower() or bool(re.search(r"\d+/\d+", c["titulo"])),
                 publicado=ya.get(c["titulo"]),
                 manual="respuesta a " in c["titulo"].lower(),
                 faltan_imagenes=[i for i in c["imagenes"] if i is None] != [],
                 se_pasa=[p["letra"] for p in c["posts"] if p["caracteres"] > 280])
        posts.append(c)
    print(json.dumps({"season": season, "week": w, "dia": dia,
                      "sin_verificar": datos.get("sin_verificar", False),
                      "aviso": "" if w == candidatas[0] else
                      f"la carpeta w{candidatas[0]:02d} no tiene nada para ese dia: esto es de la w{w:02d}",
                      "pendientes": sum(1 for p in posts if not p["publicado"]),
                      "posts": posts}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
