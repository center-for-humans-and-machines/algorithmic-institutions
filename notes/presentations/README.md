# Presentations

Figures for talks follow the style below and build on `figure_style.py`, the
base kit.

## Visual style

The deck's identity: flat, quiet, grey figures acting out the game, with red
reserved for what matters. Classy and clever, never cluttered.

**Format**
- Transparent PNG at 4× (Google Slides does not import SVG). Never draw a
  background rect; check a corner pixel has alpha 0.
- Built as SVG in Python, rendered by headless Chrome (`render()` in the kit).
- One figure fills one slide alongside its bullets: budget vertical and
  horizontal space, trim the canvas to the content.

**Palette**
| Token | Hex | Use |
|---|---|---|
| `GRAY` | `#d2d2d2` | figures in the game, props |
| `ARROW` | `#d9d9d9` | flow arrows, return lanes |
| `STACK` / `STACK_EDGE` | `#bdbdbd` / `#a8a8a8` | contribution coins; dashed loops |
| `RED` / `RED_EDGE` | `#e2502f` / `#c13a1c` | punishment, the result or rule, friction |
| `INK` | `#222222` | line work: suit, gavel, ticks, reference lines |
| `AGENT` | `#2b2b2b` | research agents only |
| `LABEL` / `SUBLABEL` | `#6b6b6b` / `#9a9a9a` | titles / captions |

A second data series next to red is a mid grey (`#7a7a7a`); confidence bands
are the line colour at opacity 0.18.

**Figures**
- Contributors: grey people, or grey robots when they are artificial humans.
- The manager: a suited figure with a gavel.
- Research agents (Claude): dark robots with white eyes. They never play the
  game. Where both robot kinds appear in one figure, add `legend_agents`
  (needs ~420 px of width).
- Robot heads alone (`robot_head`) label columns or line ends; a suited person
  marks the human/AH manager.
- Props stay flat grey shapes with white inner lines (pages, server, screen,
  shield); a single red detail marks the important element.

**Money and flow**
- Punishment: red `-$` coins. Contribution: grey `$` coins or stacks.
- Contribution flows *to* the manager; arrow width encodes amount, faded
  arrows (opacity ~0.55) a past or smaller flow.
- Process flows left to right with flat block arrows; loops are elbow
  `lane`s underneath, or dashed curves (`STACK_EDGE`, `6 5`) with the `head`
  marker for local retries.
- Friction or the key point: red, used once or twice per figure, never as
  decoration. Plain loops get no special colouring.

**Text**
- Helvetica. Step titles 17–19 px bold-ish `LABEL`; captions 13–15 px regular
  `SUBLABEL`; loop names italic `SUBLABEL`. One headline number may be large
  and red.
- Short captions, no sentences where a phrase does; every number shown must
  come from the data (no unsupported claims).

**Charts**
- No boxes, gridlines or legends where direct labels at the line ends work.
- Reference lines dashed `INK` (`7 6`); event markers lighter dotted (`3 5`).

## Base kit: `figure_style.py`

The minimum needed to reproduce the style: palette constants, `text`, `svg`
(transparent canvas with the `head` marker), `person`, `robot_head`,
`legend_agents`, `coin`, `arrow`, `lane`, and `render`.

```python
import sys; sys.path.insert(0, "notes/presentations")
from figure_style import *

body = person(60, 120, s=1.3, suit=True, gavel=True) + arrow(110, 80, 260, 80, w=7) \
     + coin(185, 80, r=11) + person(300, 120, s=1.2, robot=True)
open("fig.svg", "w").write(svg(360, 140, body))
render("fig.svg")  # fig.png, 4x, transparent
```

or render existing SVGs: `python3 notes/presentations/figure_style.py fig.svg ...`
(needs Google Chrome at its default macOS path).
