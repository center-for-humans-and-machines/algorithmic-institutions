"""Base kit for presentation figures in the deck's style (see README.md).

Palette, the figures (person, manager, robots), coins, flat arrows, labels, the
SVG wrapper and the PNG render. Scene scripts import from here; to render:

    python3 figure_style.py fig.svg [more.svg ...]   # writes fig.png at 4x, transparent
"""

import math
import re
import subprocess
import sys
import tempfile
from html import escape
from pathlib import Path

# ---- palette -----------------------------------------------------------------

GRAY = "#d2d2d2"  # figures in the game, props
ARROW = "#d9d9d9"  # flow arrows and lanes
STACK = "#bdbdbd"  # contribution coins
STACK_EDGE = "#a8a8a8"  # their rim; dashed loops
RED = "#e2502f"  # punishment, the result, friction
RED_EDGE = "#c13a1c"
INK = "#222222"  # line work: suit, gavel, ticks
AGENT = "#2b2b2b"  # research agents (Claude); never a figure in the game
LABEL = "#6b6b6b"  # titles
SUBLABEL = "#9a9a9a"  # captions
FONT = "Helvetica, Arial, sans-serif"

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# ---- text and canvas -----------------------------------------------------------


def text(x, y, lines, size=17, color=LABEL, weight=600, anchor="middle", italic=False):
    """Lines of text, the first baseline at y; XML-escaped."""
    style = ' font-style="italic"' if italic else ""
    spans = "".join(
        f'<tspan x="{x}" dy="{0 if i == 0 else size * 1.2}">{escape(ln, quote=False)}</tspan>'
        for i, ln in enumerate(lines)
    )
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}"{style}>{spans}</text>'
    )


def svg(w, h, body):
    """The canvas: no background rect, so the PNG stays transparent.
    Defines marker `head` for dashed loops (marker-end="url(#head)")."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
        f'<defs><marker id="head" viewBox="0 0 10 10" refX="7" refY="5" '
        f'markerWidth="4" markerHeight="4" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 Z" fill="{STACK_EDGE}"/></marker></defs>'
        f"{body}</svg>\n"
    )


# ---- figures -------------------------------------------------------------------


def person(x, y, s=1.0, robot=False, suit=False, gavel=False, flip=False, opacity=1.0, agent=False):
    """A figure standing with its feet on (x, y), ~65*s tall; `flip` faces it left.

    robot: an artificial human (gray). suit + gavel: the manager.
    agent: a research agent, a dark robot with white eyes."""
    sx = -s if flip else s
    body, eye = (AGENT, "white") if agent else (GRAY, INK)
    robot = robot or agent
    p = [f'<g transform="translate({x},{y}) scale({sx},{s})" opacity="{opacity}">']
    p.append(
        f'<path d="M -11 -36 L 11 -36 Q 16 -36 16.5 -31 L 18 0 L -18 0 '
        f'L -16.5 -31 Q -16 -36 -11 -36 Z" fill="{body}"/>'
    )
    if robot:
        p.append(f'<line x1="0" y1="-58" x2="0" y2="-63" stroke="{body}" stroke-width="2.5"/>')
        p.append(f'<circle cx="0" cy="-64.5" r="2.6" fill="{body}"/>')
        p.append(f'<rect x="-11.5" y="-58" width="23" height="20" rx="5" fill="{body}"/>')
        p.append(f'<circle cx="-4.6" cy="-48.5" r="2" fill="{eye}"/>')
        p.append(f'<circle cx="4.6" cy="-48.5" r="2" fill="{eye}"/>')
    else:
        p.append(f'<circle cx="0" cy="-48" r="10" fill="{GRAY}"/>')
    if suit:
        p.append(
            f'<polyline points="-6.5,-35.5 0,-25 6.5,-35.5" fill="none" '
            f'stroke="{INK}" stroke-width="1.6" stroke-linejoin="round"/>'
        )
        p.append(f'<path d="M -1.4 -34 L 1.4 -34 L 0.9 -27.5 L 0 -26 L -0.9 -27.5 Z" fill="{INK}"/>')
    if gavel:
        p.append(
            f'<line x1="12" y1="-29" x2="27" y2="-21" stroke="{GRAY}" '
            f'stroke-width="6" stroke-linecap="round"/>'
        )
        p.append(f'<rect x="28" y="-13" width="20" height="3.5" rx="1.6" fill="{GRAY}"/>')
        p.append(f'<rect x="32" y="-16" width="12" height="3" rx="1.4" fill="{GRAY}"/>')
        for cx, cy in ((33, -31), (37, -23.5)):
            p.append(
                f'<rect x="{cx - 6.5}" y="{cy - 2.3}" width="13" height="4.6" rx="2.3" '
                f'fill="white" stroke="{INK}" stroke-width="1.5" '
                f'transform="rotate(-30 {cx} {cy})"/>'
            )
    p.append("</g>")
    return "".join(p)


def robot_head(x, y, s=1.0, agent=False):
    """A robot head (with antenna) whose bottom edge sits on y: a column header or label."""
    body, eye = (AGENT, "white") if agent else (GRAY, INK)
    return (
        f'<g transform="translate({x},{y}) scale({s})">'
        f'<line x1="0" y1="-20" x2="0" y2="-25" stroke="{body}" stroke-width="2.5"/>'
        f'<circle cx="0" cy="-26.5" r="2.6" fill="{body}"/>'
        f'<rect x="-11.5" y="-20" width="23" height="20" rx="5" fill="{body}"/>'
        f'<circle cx="-4.6" cy="-10.5" r="2" fill="{eye}"/>'
        f'<circle cx="4.6" cy="-10.5" r="2" fill="{eye}"/></g>'
    )


def legend_agents(x, y):
    """Only where both kinds of robot appear: research agent dark, artificial human gray."""
    return (
        robot_head(x, y, s=0.9, agent=True)
        + text(x + 18, y - 6, ["research agent"], size=14, anchor="start")
        + robot_head(x + 160, y, s=0.9)
        + text(x + 178, y - 6, ["artificial human, in the game"], size=14, anchor="start")
    )


def coin(x, y, r=9.0, label="-$", fill=RED, edge=RED_EDGE):
    """A punishment coin; label="$", fill=STACK, edge=STACK_EDGE for a contribution."""
    return (
        f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{edge}" '
        f'stroke-width="{r * 0.2:.2f}"/>'
        f'<text x="{x}" y="{y + r * 0.34:.2f}" font-family="{FONT}" '
        f'font-weight="700" font-size="{r * 0.95:.2f}" fill="white" '
        f'text-anchor="middle">{label}</text>'
    )


# ---- flow ------------------------------------------------------------------------


def arrow(x1, y1, x2, y2, w=6.0, color=ARROW, opacity=1.0):
    """A flat block arrow from (x1, y1) to (x2, y2), head included; w encodes amount."""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    hl, hw = w * 1.9, w * 1.35
    bx, by = x2 - ux * hl, y2 - uy * hl
    pts = [
        (x1 + nx * w / 2, y1 + ny * w / 2),
        (bx + nx * w / 2, by + ny * w / 2),
        (bx + nx * hw, by + ny * hw),
        (x2, y2),
        (bx - nx * hw, by - ny * hw),
        (bx - nx * w / 2, by - ny * w / 2),
        (x1 - nx * w / 2, y1 - ny * w / 2),
    ]
    d = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    return f'<polygon points="{d}" fill="{color}" opacity="{opacity}"/>'


def lane(points, w=6, color=ARROW):
    """A flat elbow arrow through `points`, head at the last point: return paths."""
    d = "M " + " L ".join(f"{x},{y}" for x, y in points[:-1])
    (x1, y1), (x2, y2) = points[-2], points[-1]
    out = f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}" stroke-linejoin="round"/>'
    return out + arrow(x1, y1, x2, y2, w=w, color=color)


# ---- render ----------------------------------------------------------------------


def render(svg_path, scale=4):
    """SVG -> PNG next to it via headless Chrome, at `scale`x, transparent background."""
    svg_path = Path(svg_path).resolve()
    w, h = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg_path.read_text()).groups()
    page = f"<html><body style='margin:0;background:transparent'><img src='file://{svg_path}' style='width:{w}px;height:{h}px;display:block'></body></html>"
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(page)
    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
         f"--force-device-scale-factor={scale}", "--default-background-color=00000000",
         f"--window-size={w},{h}", f"--screenshot={svg_path.with_suffix('.png')}",
         f"file://{f.name}"],
        check=True, capture_output=True,
    )
    Path(f.name).unlink()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        render(p)
