# cola_posts.py — convierte los borradores de la semana en una pagina de
# copiar y pegar. Nivel 2.5: no publica nada (eso sigue siendo manual, y asi
# nos ahorramos la API de X), solo deja el trabajo hecho a un clic.
#
#   python cola_posts.py                      # ultima semana jugada
#   python cola_posts.py --season 2025 --week 18
#
# Lee  salidas/{season}/w{NN}/borradores_posts.md
# Deja salidas/{season}/w{NN}/cola_posts.html
#
# El markdown lo escribe Claude Code con el formato fijado en
# borradores_prompt.md: seccion "## ", linea "IMAGEN: fichero.png" y cada
# alternativa dentro de un bloque cercado ```post. Si el redactor se sale del
# formato, aqui se nota: la seccion sale sin tarjetas y se avisa por consola.

import argparse
import html
import io
import os
import re
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
LIMITE = 280

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
        imagen = m.group(1) if m else ""
        if imagen.lower() in ("ninguna", "none", "-", ""):
            imagen = ""
        posts = [{"letra": g.group("letra"), "texto": g.group("texto").strip()}
                 for g in RE_POST.finditer(cuerpo)]
        secciones.append({"titulo": titulo.strip(), "imagen": imagen,
                          "posts": posts})
    return banner, secciones


def tarjeta(sec, post, idx, dir_semana):
    texto = post["texto"]
    n = len(texto)
    color = OK if n <= LIMITE else MAL
    img = ""
    if sec["imagen"]:
        existe = os.path.exists(os.path.join(dir_semana, sec["imagen"]))
        if existe:
            img = (f'<img class="shot" src="{html.escape(sec["imagen"])}" '
                   f'alt="{html.escape(sec["imagen"])}">')
        else:
            img = (f'<p class="falta">No se encuentra la imagen '
                   f'<code>{html.escape(sec["imagen"])}</code> en esta carpeta</p>')
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
      {'<span class="img-nombre">' + html.escape(sec["imagen"]) + '</span>' if sec["imagen"] else ''}
    </footer>
  </article>"""


def construir(md, dir_semana, season, week):
    banner, secciones = parsear(md)
    tarjetas, idx, vacias = [], 0, []
    for sec in secciones:
        if not sec["posts"]:
            if sec["titulo"].lower().startswith(("martes", "mi", "jueves", "domingo")):
                vacias.append(sec["titulo"])
            continue
        for post in sec["posts"]:
            tarjetas.append(tarjeta(sec, post, idx, dir_semana))
            idx += 1

    aviso = ""
    if banner:
        aviso = ('<div class="banner">⛔ DATOS SIN VERIFICAR — '
                 'no publicar sin revisar la frescura de las fuentes</div>')
    if vacias:
        aviso += ('<div class="banner suave">Secciones sin borrador: '
                  + html.escape(", ".join(vacias)) +
                  ' — el redactor no las escribio (mira la nota de verificacion '
                  'en borradores_posts.md)</div>')
    if not tarjetas:
        tarjetas.append('<article class="card"><p class="falta">No se encontro '
                        'ningun borrador con el formato esperado en '
                        '<code>borradores_posts.md</code>.</p></article>')

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

    dir_semana = os.path.join(RAIZ, "salidas", str(season), f"w{int(week):02d}")
    origen = os.path.join(dir_semana, "borradores_posts.md")
    if not os.path.exists(origen):
        print(f"No hay borradores en {origen} — lanza antes el batch del martes.")
        return 1

    md = io.open(origen, encoding="utf-8").read()
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
