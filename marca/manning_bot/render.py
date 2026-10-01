"""
marca/manning_bot/render.py — mascota de Manning Bot (01-oct-2026).

Dibuja en SVG la mascota de cuerpo entero con OMAHA! y la pasa a PNG con
fondo transparente usando Chrome (playwright). La usa manning_picks.py,
pequeña al pie del PNG de los picks. Si se retoca, se retoca aqui y se relanza.
(El sello de solo la cabeza se descarto el 01-oct-2026.)

    python marca/manning_bot/render.py
"""
import os

from playwright.sync_api import sync_playwright

DIR = os.path.dirname(os.path.abspath(__file__))

# Paleta propia del bot (no es de ningun equipo)
OUT    = "#0b1420"   # contorno
STEEL  = "#5b7c9c"
STEEL2 = "#3d5a78"   # sombra
STEEL3 = "#a9c3da"   # brillo
LED    = "#ff8a1f"
LED2   = "#ffd08a"
FG     = "#EDEDED"
HAIR   = "#8a5a32"   # castaño metalizado
HAIR2  = "#b07a4a"   # brillo del pelo

DEFS = f"""
<defs>
  <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
    <feGaussianBlur stdDeviation="5" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <filter id="sticker" x="-10%" y="-10%" width="120%" height="120%">
    <feMorphology in="SourceAlpha" operator="dilate" radius="9" result="d"/>
    <feFlood flood-color="{FG}"/>
    <feComposite in2="d" operator="in" result="borde"/>
    <feMerge><feMergeNode in="borde"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <linearGradient id="metal" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{STEEL3}"/>
    <stop offset="0.35" stop-color="{STEEL}"/>
    <stop offset="1" stop-color="{STEEL2}"/>
  </linearGradient>
</defs>"""


def cabeza():
    """Cabeza en una caja de 400x370, centrada en x=200.

    Rasgos de la caricatura de Manning (referencia: su version de Los
    Simpson, sin copiar el dibujo): cara larga, frente alta, pelo castaño
    corto con raya, mirada de reojo con parpados a media asta, nariz larga,
    media sonrisa ladeada y barbilla grande. Lo de robot: metal, remaches,
    ojos LED, circuito en la frente, antena y tornillos de oreja.
    """
    s = f'stroke="{OUT}" stroke-width="8" stroke-linejoin="round"'
    remaches = "".join(
        f'<circle cx="{x}" cy="{y}" r="5" fill="{STEEL3}" stroke="{OUT}" stroke-width="3"/>'
        for x, y in [(122, 120), (278, 120), (116, 250), (284, 250)])
    circuito = f"""
    <g stroke="{LED}" stroke-width="4" fill="none" stroke-linecap="round" filter="url(#glow)">
      <path d="M150,158 L150,132 L178,132 L178,112"/>
      <path d="M200,160 L200,104"/>
      <path d="M250,158 L250,128 L224,128 L224,112"/>
    </g>
    <g fill="{LED2}" stroke="{OUT}" stroke-width="2">
      <circle cx="178" cy="112" r="5"/><circle cx="200" cy="104" r="6"/>
      <circle cx="224" cy="112" r="5"/>
    </g>"""

    def ojo(cx):
        # lente blanca, pupila LED mirando de reojo a la derecha y parpado
        # metalico a media altura (la cara de "ya se que vas a hacer")
        return f"""
    <circle cx="{cx}" cy="206" r="25" fill="{FG}" {s}/>
    <circle cx="{cx + 10}" cy="212" r="10" fill="{LED}" stroke="{OUT}" stroke-width="3" filter="url(#glow)"/>
    <path d="M{cx - 29},206 A29,29 0 0 1 {cx + 29},206 L{cx + 29},204 Q{cx},198 {cx - 29},208 Z"
          fill="{STEEL2}" {s}/>"""

    return f"""
  <g>
    <!-- antena -->
    <line x1="268" y1="62" x2="298" y2="12" stroke="{OUT}" stroke-width="12" stroke-linecap="round"/>
    <line x1="268" y1="62" x2="298" y2="12" stroke="{STEEL3}" stroke-width="5" stroke-linecap="round"/>
    <circle cx="300" cy="12" r="11" fill="{LED}" stroke="{OUT}" stroke-width="5" filter="url(#glow)"/>
    <!-- cuello -->
    <rect x="168" y="330" width="70" height="40" fill="{STEEL2}" {s}/>
    <!-- orejas: tornillos -->
    <circle cx="96" cy="222" r="24" fill="{STEEL2}" {s}/>
    <circle cx="96" cy="222" r="9" fill="{LED}" stroke="{OUT}" stroke-width="4" filter="url(#glow)"/>
    <circle cx="304" cy="222" r="24" fill="{STEEL2}" {s}/>
    <circle cx="304" cy="222" r="9" fill="{LED}" stroke="{OUT}" stroke-width="4" filter="url(#glow)"/>
    <!-- cara larga con barbilla grande, un poco adelantada -->
    <path d="M112,290 L106,120 Q106,34 200,30 Q294,34 294,120 L292,272
             Q292,338 250,362 Q222,376 188,370 Q132,356 112,290 Z" fill="url(#metal)" {s}/>
    <!-- brillo de la frente -->
    <path d="M128,150 Q130,108 160,92" stroke="{FG}" stroke-width="8" fill="none"
          stroke-linecap="round" opacity="0.5"/>
    {remaches}
    {circuito}
    <!-- pelo castaño corto con raya al lado -->
    <path d="M104,124 Q96,30 200,24 Q304,30 296,124 Q292,96 272,88 Q252,80 236,84
             L226,64 Q214,84 178,86 Q132,88 104,124 Z" fill="{HAIR}" {s}/>
    <path d="M226,64 L236,84" stroke="{OUT}" stroke-width="5"/>
    <path d="M150,52 Q190,40 222,50" stroke="{HAIR2}" stroke-width="6" fill="none" stroke-linecap="round"/>
    <!-- cejas: una algo mas alta, gesto de listillo -->
    <path d="M132,168 Q158,158 184,166" stroke="{OUT}" stroke-width="9" fill="none" stroke-linecap="round"/>
    <path d="M216,162 Q242,150 268,160" stroke="{OUT}" stroke-width="9" fill="none" stroke-linecap="round"/>
    {ojo(158)}
    {ojo(242)}
    <!-- nariz larga y recta -->
    <path d="M196,214 L208,214 L218,276 Q212,286 198,284 L188,280 Z" fill="{STEEL3}"
          stroke="{OUT}" stroke-width="5" stroke-linejoin="round"/>
    <!-- surcos de las mejillas -->
    <path d="M168,262 Q156,290 164,312" stroke="{OUT}" stroke-width="4" fill="none" stroke-linecap="round"/>
    <path d="M238,262 Q256,286 252,304" stroke="{OUT}" stroke-width="4" fill="none" stroke-linecap="round"/>
    <!-- media sonrisa ladeada: rejilla LED -->
    <path d="M160,316 Q206,330 262,300" stroke="{OUT}" stroke-width="16" fill="none" stroke-linecap="round"/>
    <path d="M166,316 Q206,328 256,302" stroke="{LED}" stroke-width="5" fill="none"
          stroke-linecap="round" filter="url(#glow)"/>
    <!-- barbilla partida -->
    <path d="M214,350 Q212,360 206,366" stroke="{OUT}" stroke-width="5" fill="none" stroke-linecap="round"/>
  </g>"""


def svg_portada():
    s = f'stroke="{OUT}" stroke-width="8" stroke-linejoin="round"'

    def brazo(puntos, ancho=58):
        d = "M" + " L".join(f"{x},{y}" for x, y in puntos)
        return (f'<path d="{d}" stroke="{OUT}" stroke-width="{ancho + 16}" fill="none" '
                f'stroke-linecap="round" stroke-linejoin="round"/>'
                f'<path d="{d}" stroke="{STEEL}" stroke-width="{ancho}" fill="none" '
                f'stroke-linecap="round" stroke-linejoin="round"/>')

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 900" width="900" height="900">
{DEFS}
<g filter="url(#sticker)">
  <!-- brazo derecho: balon en la cadera -->
  {brazo([(630, 470), (700, 600), (640, 690)])}
  <circle cx="700" cy="600" r="16" fill="{LED}" stroke="{OUT}" stroke-width="4"/>
  <!-- torso -->
  <path d="M300,440 L600,440 L640,560 L600,790 L300,790 L260,560 Z" fill="url(#metal)" {s}/>
  <path d="M300,520 L600,520" stroke="{LED}" stroke-width="10" filter="url(#glow)"/>
  <text x="450" y="730" text-anchor="middle" font-family="Arial Black, Arial, sans-serif"
        font-weight="900" font-size="190" fill="{FG}" stroke="{OUT}" stroke-width="9"
        paint-order="stroke">18</text>
  <!-- balon -->
  <g transform="rotate(-35 625 700)">
    <ellipse cx="625" cy="700" rx="62" ry="38" fill="#8a4b2a" {s}/>
    <path d="M590,700 L660,700" stroke="{FG}" stroke-width="6"/>
    <g stroke="{FG}" stroke-width="5">
      <line x1="603" y1="690" x2="603" y2="710"/><line x1="618" y1="690" x2="618" y2="710"/>
      <line x1="633" y1="690" x2="633" y2="710"/><line x1="648" y1="690" x2="648" y2="710"/>
    </g>
  </g>
  <!-- hombreras -->
  <ellipse cx="290" cy="465" rx="85" ry="48" fill="url(#metal)" {s}/>
  <ellipse cx="610" cy="465" rx="85" ry="48" fill="url(#metal)" {s}/>
  <!-- brazo izquierdo: el gesto antes del snap, señalando a la defensa -->
  {brazo([(270, 470), (175, 405), (95, 305)])}
  <circle cx="175" cy="405" r="16" fill="{LED}" stroke="{OUT}" stroke-width="4"/>
  <line x1="86" y1="292" x2="52" y2="238" stroke="{OUT}" stroke-width="34" stroke-linecap="round"/>
  <line x1="86" y1="292" x2="52" y2="238" stroke="{STEEL3}" stroke-width="20" stroke-linecap="round"/>
  <rect x="62" y="278" width="62" height="56" rx="16" fill="{STEEL}" {s} transform="rotate(-38 93 306)"/>
  <path d="M74,300 L108,322" stroke="{OUT}" stroke-width="4"/>
  <!-- cabeza -->
  <g transform="translate(250,70)">{cabeza()}</g>
  <!-- OMAHA! -->
  <path d="M590,40 L850,40 Q870,40 870,60 L870,160 Q870,180 850,180 L700,180 L635,245 L655,180
           L590,180 Q570,180 570,160 L570,60 Q570,40 590,40 Z" fill="{FG}" {s}/>
  <text x="720" y="133" text-anchor="middle" font-family="Arial Black, Arial, sans-serif"
        font-weight="900" font-size="52" fill="{LED}" stroke="{OUT}" stroke-width="4"
        paint-order="stroke">OMAHA!</text>
</g>
<!-- nombre -->
<text x="450" y="872" text-anchor="middle" font-family="Arial Black, Arial, sans-serif"
      font-weight="900" font-size="68" fill="{FG}" letter-spacing="3">MANNING <tspan fill="{LED}">BOT</tspan></text>
</svg>"""


def png(svg, destino, escala):
    with open(destino.replace(".png", ".svg"), "w", encoding="utf-8") as f:
        f.write(svg)
    with sync_playwright() as p:
        nav = p.chromium.launch(channel="chrome")
        pag = nav.new_page(device_scale_factor=escala)
        pag.set_content(f'<html><body style="margin:0;background:transparent">{svg}</body></html>')
        pag.locator("svg").screenshot(path=destino, omit_background=True)
        nav.close()
    print(f"Guardado: {destino}")


if __name__ == "__main__":
    png(svg_portada(), os.path.join(DIR, "manning_bot_portada.png"), 2)    # 1800 px
