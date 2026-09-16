"""Render the two frontier stacks as machinery (SVG in one HTML page).

Layout is authored here; every fact on the drawing (parts, badges, tints,
tooltips) comes from data/stack_parts.json. Tinted units were installed by
campaign PRs (tint = the method category, matching the progress tree);
plain outlines are stock parts; green plates are feature retrofits.

Usage:
    python scripts/data_analysis/autoresearch_summary/machinery.py
"""

import html
import json
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = Path("plots/data_analysis/autoresearch_summary/machinery.html")

CAT_COLOR = {
    "correlated-sampling": "#2a78d6",
    "structured-head": "#4a3aa7",
    "nonlinear-emission": "#eb6834",
    "architecture": "#e87ba4",
    "feature-engineering": "#1baf7a",
}
PLATE = "#1baf7a"
INK = "#1a1a19"

BAY_W, BAY_X0, BAY_GAP = 360, 30, 30
PART_H, GAP_H, TOP_Y = 62, 42, 108
ASIDE_W = 150


def esc(s):
    return html.escape(s, quote=True)


def part_box(x, y, w, part):
    """One machine part; returns svg + bottom y."""
    kind = part["kind"]
    cat = part.get("cat")
    s = []
    if part.get("tip"):
        s.append(f"<title>{esc(part['tip'])}</title>")
    if kind == "unit":
        color = CAT_COLOR[cat]
        s.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{PART_H}" rx="6" '
            f'fill="{color}" fill-opacity="0.14" stroke="{color}"/>'
        )
    else:
        s.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{PART_H}" rx="6" '
            f'fill="none" stroke="currentColor" stroke-opacity="0.6"/>'
        )
    cx = x + w / 2
    s.append(
        f'<text x="{cx}" y="{y + 25}" text-anchor="middle" font-size="12" '
        f'fill="currentColor" font-weight="600">{esc(part["label"])}</text>'
    )
    s.append(
        f'<text x="{cx}" y="{y + 42}" text-anchor="middle" font-size="10" '
        f'fill="currentColor" opacity="0.75">{esc(part["sub"])}</text>'
    )
    bottom = y + PART_H
    if part.get("badge"):
        s.append(
            f'<rect x="{x + w - 44}" y="{y - 8}" width="38" height="17" '
            f'rx="8" fill="{INK}"/>'
            f'<text x="{x + w - 25}" y="{y + 4.5}" fill="#fcfcfb" '
            f'text-anchor="middle" font-size="10">{esc(part["badge"])}</text>'
        )
    for plate in part.get("plates", []):
        pw = 140
        s.append(
            f'<rect x="{x + w - pw - 10}" y="{bottom - 9}" width="{pw}" '
            f'height="18" rx="4" fill="{PLATE}" fill-opacity="0.2" '
            f'stroke="{PLATE}"/>'
            f'<text x="{x + w - pw / 2 - 10}" y="{bottom + 4}" '
            f'text-anchor="middle" font-size="9.5" fill="currentColor">'
            f'{esc(plate["label"])}  {esc(plate["badge"])}</text>'
        )
        bottom += 9
    return f'<g>{"".join(s)}</g>', bottom


def render_bay(bx, bay, chassis_y):
    w, cx = BAY_W - 80, bx + BAY_W / 2
    x = bx + 40
    s = [
        f'<rect x="{bx}" y="60" width="{BAY_W}" height="{chassis_y - 90}" '
        f'rx="10" fill="none" stroke="currentColor" stroke-opacity="0.35"/>',
        f'<text x="{cx}" y="84" text-anchor="middle" font-size="13" '
        f'font-weight="600" fill="currentColor">{esc(bay["name"])}</text>',
    ]
    if bay["note"]:
        s.append(
            f'<text x="{cx}" y="99" text-anchor="middle" font-size="10" '
            f'fill="currentColor" opacity="0.7">{esc(bay["note"])}</text>'
        )
    y = TOP_Y
    prev_px = cx
    for i, part in enumerate(bay["parts"]):
        px = (bx + 12 + ASIDE_W + 18 + (BAY_W - ASIDE_W - 42) / 2
              if part.get("aside") else cx)
        if i:
            arrow = part.get("arrow", {})
            marker = ' marker-start="url(#arr)"' if arrow.get("two_way") else ""
            s.append(
                f'<line x1="{prev_px}" y1="{y}" x2="{px}" '
                f'y2="{y + GAP_H - 4}" '
                f'stroke="currentColor" marker-end="url(#arr)"{marker}/>'
            )
            if arrow.get("label"):
                s.append(
                    f'<text x="{max(prev_px, px) + 8}" '
                    f'y="{y + GAP_H / 2 + 3}" '
                    f'font-size="10" fill="currentColor" opacity="0.75">'
                    f'{esc(arrow["label"])}</text>'
                )
            y += GAP_H
        prev_px = px
        if part.get("aside"):
            a = part["aside"]
            ax = bx + 12
            s.append(
                f'<rect x="{ax}" y="{y + 5}" width="{ASIDE_W}" height="52" '
                f'rx="6" fill="none" stroke="currentColor" '
                f'stroke-opacity="0.4" stroke-dasharray="5 4"/>'
                f'<text x="{ax + ASIDE_W / 2}" y="{y + 27}" '
                f'text-anchor="middle" font-size="10" fill="currentColor" '
                f'opacity="0.6">{esc(a["label"])}</text>'
                f'<text x="{ax + ASIDE_W / 2}" y="{y + 43}" '
                f'text-anchor="middle" font-size="8.5" fill="currentColor" '
                f'opacity="0.6">{esc(a["sub"])}</text>'
            )
            box, y = part_box(ax + ASIDE_W + 18, y,
                              BAY_W - ASIDE_W - 42, part)
        else:
            box, y = part_box(x, y, w, part)
        s.append(box)
    s.append(
        f'<line x1="{prev_px}" y1="{y}" x2="{cx}" y2="{chassis_y - 5}" '
        f'stroke="currentColor" marker-end="url(#arr)"/>'
        f'<text x="{cx + 8}" y="{(y + chassis_y) / 2}" font-size="10.5" '
        f'fill="currentColor">{esc(bay["out_label"])}</text>'
        f'<circle cx="{cx}" cy="{chassis_y}" r="3.5" fill="currentColor"/>'
    )
    return "".join(s)


def render_machine(machine):
    slots = max(
        len(b["parts"]) + sum(len(p.get("plates", [])) for p in b["parts"])
        for b in machine["bays"]
    )
    chassis_y = TOP_Y + slots * PART_H + (slots - 1) * GAP_H + 60
    height = chassis_y + 90
    width = BAY_X0 * 2 + BAY_W * 3 + BAY_GAP * 2
    bays = "".join(
        render_bay(BAY_X0 + i * (BAY_W + BAY_GAP), bay, chassis_y)
        for i, bay in enumerate(machine["bays"])
    )
    conveyor = (
        f'<line x1="{BAY_X0}" y1="{chassis_y}" x2="{width - BAY_X0}" '
        f'y2="{chassis_y}" stroke="currentColor" stroke-width="1.6" '
        f'marker-end="url(#arr)"/>'
        f'<text x="{width / 2}" y="{chassis_y + 22}" text-anchor="middle" '
        f'font-size="11" fill="currentColor">round state: contributions '
        f'-&gt; common good -&gt; punishments</text>'
        f'<path d="M {width - BAY_X0} {chassis_y + 35} L {BAY_X0} '
        f'{chassis_y + 35}" stroke="currentColor" stroke-dasharray="6 5" '
        f'stroke-opacity="0.6" fill="none" marker-end="url(#arr)"/>'
        f'<text x="{width / 2}" y="{chassis_y + 55}" text-anchor="middle" '
        f'font-size="10.5" fill="currentColor" opacity="0.75">feeds the '
        f'next round (24 rounds x 100 episodes, seed 42)</text>'
    )
    return (
        f'<h2>{esc(machine["title"])}</h2>\n<figure>\n'
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'font-family="system-ui, sans-serif" '
        f'aria-label="{esc(machine["title"])} drawn as a three-bay machine.">'
        f'<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>'
        f"{bays}{conveyor}</svg>\n</figure>\n"
    )


LEGEND = """
<p class="legend">
<span style="color:#e87ba4">&#9632;</span> architecture add-on &nbsp;
<span style="color:#eb6834">&#9632;</span> nonlinear trunk &nbsp;
<span style="color:#4a3aa7">&#9632;</span> structured head &nbsp;
<span style="color:#2a78d6">&#9632;</span> correlated-sampling unit &nbsp;
<span style="color:#1baf7a">&#9632;</span> feature retrofit plate &nbsp;
&#9633; stock part &nbsp; &#11044;<small>#PR</small> installed by
&mdash; hover a tinted part for its story.
</p>
"""


def main():
    machines = json.loads((DATA / "stack_parts.json").read_text())["machines"]
    body = "\n".join(render_machine(m) for m in machines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "<title>Stack Machinery</title>\n<style>\n"
        "body { background: #fcfcfb; color: #1a1a19; font-family: system-ui,"
        " sans-serif; margin: 2rem auto; max-width: 1240px; }\n"
        "svg { max-width: 100%; height: auto; }\n"
        "figure { margin: 0 0 2.5rem 0; }\n"
        ".legend { color: #52514e; font-size: 0.9rem; }\n"
        "</style>\n" + LEGEND + body
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
