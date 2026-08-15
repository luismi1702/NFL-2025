"""
lab/codemod_logos.py — Auditoría visual jul-2026.
Sustituye la heurística de zoom por aspect-ratio + HARD_PENALTY por
normalización por píxeles reales: zoom = base * 500 / max(h, w).
Así todos los logos ocupan el mismo lienzo aunque el PNG tenga otra
resolución (NYJ es 4096px; el resto 500px).

Ejecutar desde la raíz:  python lab/codemod_logos.py
"""
import re
import glob

PATTERNS = [
    # A) bloque estándar con base_zoom
    (re.compile(
        r"[ \t]*if team in HARD_PENALTY:\n"
        r"[ \t]*zoom = base_zoom / HARD_PENALTY\[team\]\n"
        r"[ \t]*else:\n"
        r"[ \t]*div = np\.clip\(1\.0 \+ 0\.6 \* max\(0\.0, aspect - 1\.3\), 1\.0, 2\.2\)\n"
        r"[ \t]*zoom = base_zoom / div\n"),
     "        zoom = base_zoom * 500.0 / max(h, w)   # normaliza por pixeles reales\n"),
    # B) variante ranking_* (param zoom, var z)
    (re.compile(
        r"[ \t]*if team in HARD_PENALTY:\n"
        r"[ \t]*z = zoom / HARD_PENALTY\[team\]\n"
        r"[ \t]*else:\n"
        r"[ \t]*div = np\.clip\(1\.0 \+ 0\.6 \* max\(0\.0, aspect - 1\.3\), 1\.0, 2\.2\)\n"
        r"[ \t]*z = zoom / div\n"),
     "        z = zoom * 500.0 / max(h, w)   # normaliza por pixeles reales\n"),
    # C) one-liner con backslash (informe, matchup, tendencias, clutch, game_script)
    (re.compile(
        r"[ \t]*zoom = base_zoom / HARD_PENALTY\[team\] if team in HARD_PENALTY else \\\n"
        r"[ \t]*base_zoom / np\.clip\(1\.0 \+ 0\.6 \* max\(0\.0, aspect - 1\.3\), 1\.0, 2\.2\)\n"),
     "        zoom = base_zoom * 500.0 / max(h, w)   # normaliza por pixeles reales\n"),
    # D) contenders_tracker
    (re.compile(
        r"[ \t]*div = HARD_PENALTY\.get\(team, np\.clip\(1\.0 \+ 0\.6 \* max\(0\.0, aspect - 1\.3\), 1\.0, 2\.2\)\)\n"
        r"[ \t]*return OffsetImage\(img, zoom=zoom / div, resample=True\)\n"),
     "        return OffsetImage(img, zoom=zoom * 500.0 / max(h, w), resample=True)\n"),
    # E) RankingEPAadjustado (inline, minúsculas)
    (re.compile(
        r"[ \t]*if team in hard_penalty:\n"
        r"[ \t]*zoom = base_zoom / hard_penalty\[team\]\n"
        r"[ \t]*else:\n"
        r"[ \t]*if aspect <= 1\.3:\n"
        r"[ \t]*divisor = 1\.0\n"
        r"[ \t]*else:\n"
        r"[ \t]*divisor = 1\.0 \+ 0\.6 \* \(aspect - 1\.3\)\n"
        r"[ \t]*divisor = np\.clip\(divisor, 1\.0, 2\.2\)\n"
        r"[ \t]*zoom = base_zoom / divisor\n"),
     "            zoom = base_zoom * 500.0 / max(h, w)   # normaliza por pixeles reales\n"),
]


def main():
    tocados, pendientes = [], []
    for f in sorted(glob.glob("*.py")):
        src = open(f, encoding="utf-8").read()
        nuevo = src
        for pat, rep in PATTERNS:
            nuevo = pat.sub(rep, nuevo)
        if nuevo != src:
            open(f, "w", encoding="utf-8").write(nuevo)
            tocados.append(f)
        # aviso si sigue usando HARD_PENALTY para calcular zoom
        resto = [l for l in nuevo.splitlines()
                 if "HARD_PENALTY" in l and "=" in l and "{" not in l and "get(" not in l
                 and not l.strip().startswith("#")]
        if any("zoom" in l or "div" in l or " z " in l for l in resto):
            pendientes.append(f)
    print("Modificados:", len(tocados))
    for f in tocados:
        print("  ", f)
    if pendientes:
        print("REVISAR A MANO (siguen usando HARD_PENALTY en zoom):")
        for f in pendientes:
            print("  ", f)


if __name__ == "__main__":
    main()
