# cola_posts.py — convierte los borradores de la semana en una pagina de
# copiar y pegar. Nivel 2.5: no publica nada (eso sigue siendo manual, y asi
# nos ahorramos la API de X), solo deja el trabajo hecho a un clic.
#
#   python cola_posts.py                      # ultima semana jugada
#   python cola_posts.py --season 2025 --week 18
#
# Lee  salidas/{season}/w{NN}/textos/borradores_lunes.md y borradores_posts.md
# Deja salidas/{season}/w{NN}/cola_posts.html (arriba del todo de la semana)
# Las imagenes se buscan por nombre en los cajones previas/, partidos/ y
# liga/ de esta semana y de la siguiente, y se enlazan con ruta relativa.
#
# El markdown lo escribe Claude Code con el formato fijado en
# borradores_prompt.md: seccion "## ", linea "IMAGEN: fichero.png" y cada
# alternativa dentro de un bloque cercado ```post. IMAGEN admite varias
# separadas por comas. Si el redactor se sale del
# formato, aqui se nota: la seccion sale sin tarjetas y se avisa por consola.

import argparse
import html
import io
import os
import re
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
LIMITE = 280
# Orden en la pagina de los que existan; cualquier otro borradores_*.md entra detras
ORIGENES = ["borradores_lunes.md", "borradores_posts.md", "borradores_domingo.md"]

BG, CARD, FG, GRID, ACCENT = "#0f1115", "#151924", "#EDEDED", "#2a2f3a", "#2d6cdf"
OK, AVISO, MAL = "#06d6a0", "#ffd166", "#d84a4a"

RE_POST = re.compile(
    r"^\*\*\[(?P<letra>[A-Z])\]\*\*[^\n]*\n+```post\r?\n(?P<texto>.*?)\r?\n```",
    re.MULTILINE | re.DOTALL)


def parsear(md):
    """Devuelve (banner, [ {titulo, imagen, posts:[{letra, texto}]} ])."""
    banner = "DATOS SIN VERIFICAR" in md.split("\n", 3)[0].upper() or \
             md.lstrip().startswith("⛔")
    secciones = []
    # split conservando el titulo de cada seccion "## ..."
    trozos = re.split(r"^## +", md, flags=re.MULTILINE)[1:]
    for trozo in trozos:
        titulo, _, cuerpo = trozo.partition("\n")
        m = re.search(r"^IMAGEN:\s*(.+?)\s*$", cuerpo, re.MULTILINE)
        # Una o varias imagenes separadas por comas (X admite hasta 4): los
        # posts de partido llevan resumen + ficha, en ese orden
        imagen = m.group(1) if m else ""
        if imagen.lower() in ("ninguna", "none", "-", ""):
            imagen = ""
        imagenes = [i.strip() for i in imagen.split(",") if i.strip()]
        posts = [{"letra": g.group("letra"), "texto": g.group("texto").strip()}
                 for g in RE_POST.finditer(cuerpo)]
        secciones.append({"titulo": titulo.strip(), "imagenes": imagenes,
                          "posts": posts})
    return banner, secciones


def indexar(carpetas, dir_html):
    """{nombre de fichero en minusculas: src relativo a la pagina}.

    Desde el 21-sep-2026 los PNG viven en cajones (previas/, partidos/,
    liga/) y los de viernes, sabado y domingo estan ademas en la carpeta de
    la jornada SIGUIENTE, asi que la cola busca en las dos semanas y en
    todos los cajones, y escribe la ruta relativa que necesita el <img>.
    """
    idx = {}
    for carpeta in carpetas:
        for raiz, _, ficheros in os.walk(carpeta):
            for f in ficheros:
                idx.setdefault(f.lower(), os.path.relpath(
                    os.path.join(raiz, f), dir_html).replace(os.sep, "/"))
    return idx


def tarjeta(sec, post, idx, imagenes):
    texto = post["texto"]
    n = len(texto)
    color = OK if n <= LIMITE else MAL
    img = ""
    for nombre in sec["imagenes"]:
        src = imagenes.get(nombre.lower())
        if src:
            img += (f'<img class="shot" src="{html.escape(src)}" '
                    f'alt="{html.escape(nombre)}">')
        else:
            img += (f'<p class="falta">No se encuentra la imagen '
                    f'<code>{html.escape(nombre)}</code> en la semana</p>')
    return f"""
  <article class="card">
    <header>
      <h2>{html.escape(sec["titulo"])} <span class="letra">[{post["letra"]}]</span></h2>
      <button class="copiar" data-id="p{idx}">Copiar</button>
    </header>
    {img}
    <pre class="post" id="p{idx}">{html.escape(texto)}</pre>
    <footer>
      <span class="chars" style="color:{color}">{n}/{LIMITE} caracteres</span>
      {'<span class="img-nombre">' + html.escape(" · ".join(sec["imagenes"])) + '</span>' if sec["imagenes"] else ''}
    </footer>
  </article>"""


# La semana de publicacion de una jornada, en el orden del calendario
# (docs/calendario-posts.md; Luis, 27-sep-2026): empieza el viernes con el
# analisis del TNF y acaba el jueves con el bot, que ya mira a la siguiente.
# Cada dia sale SIEMPRE: con sus posts o con una tarjeta PENDIENTE.
DIAS = [(("viernes",), "VIERNES"), (("sábado", "sabado"), "SÁBADO"),
        (("domingo",), "DOMINGO"), (("lunes",), "LUNES"),
        (("martes",), "MARTES"), (("mi",), "MIÉRCOLES"), (("jueves",), "JUEVES")]
VIERNES, SABADO, DOMINGO, LUNES = 0, 1, 2, 3

# Que toca cada dia (calendario-posts.md) y que PNG de la carpeta lo ilustran
CALENDARIO = [
    "Análisis del TNF jugado el jueves: resumen + ficha del partido 01 (manual).",
    "Pieza de DUELO: un partido del domingo a fondo, en hilo (manual, lab/). "
    "Ese partido no abre el hilo de previas.",
    "HILO de la jornada: una previa por partido, abre el gordo. PNG en previas/; "
    "el texto va en textos/borradores_domingo.md.",
    "Un post por partido jugado, sin el MNF (batch lunes 10:00).",
    "Dato de la semana + Monday Night, dos posts, uno por equipo (batch martes 8:00).",
    "Power Rankings + hilo de MVPs de la jornada (batch martes).",
    "Bot: balance de la jornada + picks de la siguiente, gancho previa del TNF (batch martes).",
]
PNG_DEL_DIA = [r"^01_(ficha|resumen)_", r"^duelo_", None, None,
               r"^dato_semana", r"^power_rankings", None]


def dia_de(sec):
    """Indice del dia de publicacion segun el titulo. Las tarjetas 'SIN POST'
    son partidos: el 01 es el TNF (viernes) y el resto van con el lunes."""
    if "dia" in sec:
        return sec["dia"]
    if sec.get("hueco"):
        return VIERNES if sec["imagenes"][0].startswith("01_") else LUNES
    t = sec["titulo"].lower()
    return next((k for k, (d, _) in enumerate(DIAS) if t.startswith(d)), len(DIAS))


def orden(indexada):
    """Agrupada por DIA de publicacion (pedido por Luis el 15-sep-2026) y,
    dentro de cada dia, en el orden de la carpeta de la semana (por nombre de
    fichero: 01_, 02_... por kickoff, luego dato_semana, power_rankings...).
    Los posts sin imagen van al final de su dia, en el orden del markdown."""
    i, sec = indexada
    # Un hilo se publica en el orden del markdown (abre el partido gordo, no
    # el 01_ del kickoff)
    if "hilo" in sec["titulo"].lower():
        return (dia_de(sec), 0, "", i)
    if not sec["imagenes"]:
        return (dia_de(sec), 1, "", i)
    return (dia_de(sec), 0, sec["imagenes"][0].lower(), i)


def construir(md, dir_semana, season, week, imagenes=None):
    if imagenes is None:
        imagenes = indexar([dir_semana], dir_semana)
    banner, secciones = parsear(md)
    for sec in secciones:
        # tambien dentro del post: 01_ficha antes que 01_resumen, como en la carpeta
        sec["imagenes"] = sorted(sec["imagenes"], key=str.lower)
    # Todo partido de la carpeta tiene tarjeta, aunque nadie le haya escrito
    # post: la cola es un espejo de la carpeta (el 15-sep-2026 faltaba el
    # BAL@IND por estar ya publicado y no habia forma de saberlo mirando)
    usadas = {i.lower() for s in secciones for i in s["imagenes"]}
    for f in sorted((os.path.basename(v) for v in imagenes.values()), key=str.lower):
        m = re.match(r"(\d\d)_ficha_([A-Z]+)_vs_([A-Z]+)_", f)
        if not m or f.lower() in usadas:
            continue
        resumen = f.replace("_ficha_", "_resumen_")
        titulo = (f"Análisis del TNF — {m.group(2)} vs {m.group(3)}"
                  if m.group(1) == "01" else f"SIN POST — {m.group(2)} vs {m.group(3)}")
        secciones.append({"titulo": titulo,
                          "imagenes": [f, resumen], "posts": [], "hueco": True})
    # Cada dia del calendario tiene al menos una tarjeta: si nadie ha escrito
    # su post, sale PENDIENTE con lo que toca y los PNG que ya haya
    ocupados = {dia_de(s) for s in secciones if s["posts"] or s.get("hueco")}
    for k, (_, nombre) in enumerate(DIAS):
        if k in ocupados:
            continue
        pat = PNG_DEL_DIA[k]
        fotos = sorted((os.path.basename(v) for v in imagenes.values()
                        if pat and re.match(pat, os.path.basename(v))
                        and v.lower().endswith(".png")), key=str.lower)
        secciones.append({"titulo": f"{nombre} — PENDIENTE", "imagenes": fotos,
                          "posts": [], "hueco": True, "dia": k,
                          "nota": CALENDARIO[k]})
    secciones = [s for _, s in sorted(enumerate(secciones), key=orden)]
    tarjetas, idx, vacias = [], 0, []
    dia_actual = None
    n_tarjetas = 0
    for sec in secciones:
        # El rotulo del dia solo sale si ese dia tiene alguna tarjeta (el jueves
        # sin post del bot no deja un titulo vacio)
        d = dia_de(sec)
        if d != dia_actual and (sec.get("hueco") or sec["posts"]):
            dia_actual = d
            nombre = DIAS[d][1] if d < len(DIAS) else "OTROS"
            tarjetas.append(f'<h2 class="dia">{nombre}</h2>')
        if sec.get("hueco"):
            n_tarjetas += 1
            fotos = "".join(
                f'<img class="shot" src="{html.escape(imagenes[n.lower()])}" '
                f'alt="{html.escape(n)}">'
                for n in sec["imagenes"] if n.lower() in imagenes)
            tarjetas.append(f"""
  <article class="card">
    <header><h2>{html.escape(sec["titulo"])}</h2></header>
    {fotos}
    <p class="falta">{html.escape(sec.get("nota", "Este partido no tiene post en los borradores."))}</p>
  </article>""")
            continue
        if not sec["posts"]:
            if dia_de(sec) < len(DIAS):
                vacias.append(sec["titulo"])
            continue
        for post in sec["posts"]:
            tarjetas.append(tarjeta(sec, post, idx, imagenes))
            idx += 1
            n_tarjetas += 1

    aviso = ""
    if banner:
        aviso = ('<div class="banner">⛔ DATOS SIN VERIFICAR — '
                 'no publicar sin revisar la frescura de las fuentes</div>')
    if vacias:
        aviso += ('<div class="banner suave">Secciones sin borrador: '
                  + html.escape(", ".join(vacias)) +
                  ' — el redactor no las escribio (mira la nota de verificacion '
                  'en los borradores)</div>')
    if not n_tarjetas:
        tarjetas.append('<article class="card"><p class="falta">No se encontro '
                        'ningun borrador con el formato esperado en '
                        '<code>borradores_lunes.md</code> ni <code>borradores_posts.md</code>.</p></article>')

    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cola de posts — NFL {season} semana {week}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:24px; background:{BG}; color:{FG};
         font-family:"Segoe UI",system-ui,sans-serif; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .sub {{ color:#888; font-size:13px; margin:0 0 20px; }}
  .banner {{ background:{MAL}; color:#1a1a1a; font-weight:600; padding:10px 14px;
             border-radius:8px; margin-bottom:16px; }}
  .banner.suave {{ background:{AVISO}; }}
  .grid {{ display:grid; gap:16px; grid-template-columns:repeat(auto-fill,minmax(380px,1fr)); }}
  .dia {{ grid-column:1/-1; margin:14px 0 0; padding-bottom:6px; font-size:16px;
          letter-spacing:.08em; color:{ACCENT}; border-bottom:1px solid {GRID}; }}
  .card {{ background:{CARD}; border:1px solid {GRID}; border-radius:10px; padding:14px; }}
  .card header {{ display:flex; align-items:center; justify-content:space-between; gap:10px; }}
  .card h2 {{ font-size:14px; margin:0; font-weight:600; }}
  .letra {{ color:{ACCENT}; }}
  .copiar {{ background:{ACCENT}; color:#fff; border:0; border-radius:6px;
             padding:6px 14px; font-size:13px; cursor:pointer; flex:0 0 auto; }}
  .copiar:hover {{ filter:brightness(1.15); }}
  .copiar.hecho {{ background:{OK}; color:#10241d; }}
  .copiar.manual {{ background:{AVISO}; color:#2a2410; }}
  .post::selection, .post ::selection {{ background:{ACCENT}; color:#fff; }}
  .shot + .shot {{ margin-top:8px; }}
  .shot {{ display:block; width:100%; height:auto; border-radius:6px;
           margin:12px 0; border:1px solid {GRID}; }}
  .post {{ white-space:pre-wrap; word-wrap:break-word; font-family:inherit;
           font-size:14px; line-height:1.45; background:{BG}; border:1px solid {GRID};
           border-radius:6px; padding:12px; margin:12px 0 8px; }}
  .card footer {{ display:flex; justify-content:space-between; gap:10px;
                  font-size:12px; color:#888; }}
  .chars {{ font-weight:600; }}
  .img-nombre {{ font-family:ui-monospace,monospace; }}
  .falta {{ color:{AVISO}; font-size:13px; }}
  .pie {{ color:#888; font-size:12px; margin-top:24px; font-style:italic; }}
</style>
</head>
<body>
<h1>Cola de posts — NFL {season}, semana {week}</h1>
<p class="sub">Copiar, pegar en X y arrastrar la imagen desde esta misma carpeta.
Los borradores son borradores: la verificacion triple sigue siendo tuya.</p>
{aviso}
<div class="grid">{"".join(tarjetas)}
</div>
<p class="pie">Generado por cola_posts.py desde borradores_posts.md — @CuartayDato</p>
<script>
document.querySelectorAll(".copiar").forEach(function (b) {{
  b.addEventListener("click", function () {{
    var t = document.getElementById(b.dataset.id).textContent;
    var fin = function () {{
      var antes = b.textContent;
      b.textContent = "Copiado";
      b.classList.add("hecho");
      setTimeout(function () {{ b.textContent = antes; b.classList.remove("hecho"); }}, 1400);
    }};
    if (navigator.clipboard && navigator.clipboard.writeText) {{
      navigator.clipboard.writeText(t).then(fin, function () {{ manual(t, fin, b); }});
    }} else {{ manual(t, fin, b); }}
  }});
}});
function manual(t, fin, b) {{
  // Abierto con doble clic (file://) no hay navigator.clipboard: se copia con
  // execCommand y, si tampoco puede, se selecciona el texto para un Ctrl+C.
  var a = document.createElement("textarea");
  a.value = t;
  a.style.cssText = "position:fixed;top:-1000px;left:-1000px";
  document.body.appendChild(a);
  a.focus(); a.select();
  var ok = false;
  try {{ ok = document.execCommand("copy"); }} catch (e) {{ ok = false; }}
  document.body.removeChild(a);
  if (ok) {{ fin(); return; }}
  seleccionar(b);
}}
function seleccionar(b) {{
  // Ultimo recurso: dejar el post seleccionado y decirlo.
  var pre = document.getElementById(b.dataset.id);
  var r = document.createRange();
  r.selectNodeContents(pre);
  var sel = window.getSelection();
  sel.removeAllRanges(); sel.addRange(r);
  var antes = b.textContent;
  b.textContent = "Pulsa Ctrl+C";
  b.classList.add("manual");
  setTimeout(function () {{ b.textContent = antes; b.classList.remove("manual"); }}, 2500);
}}
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(
        description="Pagina de copiar y pegar a partir de borradores_posts.md",
        add_help=True)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--raiz", action="store_true", help=argparse.SUPPRESS)
    args, _ = ap.parse_known_args()

    season, week = args.season, args.week
    if season is None or week is None:
        from pbp_loader import ultima_semana, temporada_actual
        season = season if season is not None else temporada_actual()
        week = week if week is not None else ultima_semana(season)
    if not week:
        print("Sin semana jugada: nada que preparar.")
        return 1

    def carpeta(w):
        return os.path.join(RAIZ, "salidas", str(season), f"w{int(w):02d}")

    dir_semana = carpeta(week)
    # UNA cola por semana, en su carpeta y con SOLO lo de su carpeta (Luis,
    # 27-sep-2026): w03/cola_posts.html lleva viernes, sabado y domingo de la
    # jornada 3 (TNF, duelo, hilo de previas) y lunes a jueves despues de
    # jugarla. Entra cualquier textos/borradores_*.md (lunes, posts, domingo,
    # los que vengan) menos las copias _previo. La ruta plana sigue valiendo
    # para las semanas archivadas antes del 21-sep-2026.
    origenes = []
    for sub in ("textos", ""):
        d = os.path.join(dir_semana, sub)
        if not os.path.isdir(d):
            continue
        propios = sorted(f for f in os.listdir(d)
                         if f.startswith("borradores_") and f.endswith(".md")
                         and "_previo" not in f and "_prompt" not in f)
        # lunes y posts primero, como siempre; el resto detras
        propios.sort(key=lambda f: (ORIGENES.index(f) if f in ORIGENES else 99, f))
        origenes += [os.path.join(d, f) for f in propios]
    if not origenes:
        # Sin textos todavia la pagina se genera igual: es el espejo de la
        # carpeta y cada partido sale con su tarjeta SIN POST
        print(f"Sin borradores en {dir_semana}: pagina solo con las imagenes.")

    trozos = [io.open(o, encoding="utf-8").read() for o in origenes]
    # El banner de datos sin verificar de cualquiera de los dos va arriba
    banner = any(t.lstrip().startswith("⛔") for t in trozos)
    md = ("⛔ DATOS SIN VERIFICAR" + chr(10) if banner else "") + (2 * chr(10)).join(trozos)
    print("Borradores: " + ", ".join(
        os.path.join(os.path.basename(os.path.dirname(o)), os.path.basename(o))
        for o in origenes))
    destino = os.path.join(dir_semana, "cola_posts.html")
    io.open(destino, "w", encoding="utf-8", newline="\n").write(
        construir(md, dir_semana, season, int(week)))

    banner, secciones = parsear(md)
    n = sum(len(s["posts"]) for s in secciones)
    print(f"{n} borradores -> {destino}")
    largos = [(s["titulo"], p["letra"], len(p["texto"]))
              for s in secciones for p in s["posts"] if len(p["texto"]) > LIMITE]
    for titulo, letra, n_ in largos:
        print(f"  AVISO: {titulo} [{letra}] se pasa de 280: {n_} caracteres")
    if banner:
        print("  AVISO: el markdown trae el banner de datos sin verificar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
