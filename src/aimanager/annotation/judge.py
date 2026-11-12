"""Rule judge: which codes of a codebook does a rule implement.

Per rule and code the judge returns `evidence` (the verbatim line), `present`
and `switchable_off`, in that order. The judge runs `judge.repeats` times, each
repeat with the codes in a different seeded order (repeat 0 in codebook order),
so the answers do not depend on where a code sits in the list. Then:

- a `present: true` answer with empty evidence is re-asked once (same prompt,
  schema restricted to those codes); if it stays empty, it is recorded as
  `present: false` with `unsupported: true`;
- per rule and code the final answer is the unanimous answer of the repeats or,
  when they disagree, the answer of one tie-break run in a further order;
- repeats 0 and 1 give the judge's test-retest reliability per code.
"""

import csv
import io
import random

from . import prompting as P
from .client import Call
from .common import atomic_write_text, check_failures, log, quote_in_source
from .expert import kappa_stats
from .schemas import judge_answers, judge_model

TIEBREAK = "tiebreak"


def code_order(codes, repeat, seed):
    """Codes for one repeat: codebook order for repeat 0, else a seeded shuffle of
    the families and of the codes within each family (families stay together)."""
    if repeat == 0:
        return list(codes)
    rng = random.Random(f"{seed}:{repeat}")
    fams = list(dict.fromkeys(c.get("family") for c in codes))
    rng.shuffle(fams)
    out = []
    for fam in fams:
        group = [c for c in codes if c.get("family") == fam]
        rng.shuffle(group)
        out.extend(group)
    return out


def _answer(j):
    return {
        "evidence": j.evidence,
        "present": j.present,
        "switchable_off": j.switchable_off,
        "unsupported": False,
        "reasked": False,
    }


def _unsupported(a):
    return a["present"] and not a["evidence"].strip()


def resolve(answers, tiebreak=None):
    """(final answer, used tie-break) for one rule and code.

    `answers` holds one answer per repeat; the final answer is the first repeat's
    when all agree on `present`, else the tie-break answer.
    """
    if len({a["present"] for a in answers}) == 1:
        return answers[0], False
    if tiebreak is None:
        raise ValueError("repeats disagree and there is no tie-break answer")
    return tiebreak, True


class _Judge:
    def __init__(self, client, cfg, step, codes, versions, families=None):
        self.client, self.cfg, self.step = client, cfg, step
        self.codes, self.families = codes, families
        self.versions = {v["stem"]: v for v in versions}
        self.instr = P.load("judge")
        self.seed = cfg.get("sampling", {}).get("seed", 0)
        self.shared = {}
        self.failed = []

    def _shared(self, order_key, ordered):
        if order_key not in self.shared:
            self.shared[order_key] = P.render_codebook(ordered, families=self.families)
        return self.shared[order_key]

    def _call(self, stem, tag, ordered, names, reask=False):
        v = self.versions[stem]
        shared = self._shared(tag, ordered)
        return Call(
            self.step,
            f"{stem}#{tag}" + ("#reask" if reask else ""),
            P.messages(self.instr, v["rule"], shared, context=v["setting"]),
            judge_model(names),
            "judge",
            meta={
                "codes": names,
                "doc": v["rule"],
                "stem": stem,
                "repeat": tag,
                "reask": reask,
            },
        )

    def run(self, jobs):
        """jobs: [(stem, tag, ordered codes, names to ask)] -> {(stem, tag): answers}.

        Answers with present and empty evidence are re-asked once.
        """
        calls = [self._call(s, t, o, n) for s, t, o, n in jobs]
        info = {(s, t): (o, n) for s, t, o, n in jobs}
        out = {}
        for res in self.client.map(calls, label=f"{self.step} judge"):
            stem, tag = res.call.meta["stem"], res.call.meta["repeat"]
            if not res.ok:
                self.failed.append((res.call.item_id, res.error))
                continue
            names = res.call.meta["codes"]
            out[(stem, tag)] = {
                n: _answer(j) for n, j in judge_answers(res.value, names).items()
            }
        reask = []
        for (stem, tag), ans in out.items():
            names = [n for n, a in ans.items() if _unsupported(a)]
            if names:
                ordered = info[(stem, tag)][0]
                order = [c["name"] for c in ordered if c["name"] in names]
                reask.append(self._call(stem, tag, ordered, order, reask=True))
        if reask:
            log(f"  {self.step}: re-asking {len(reask)} replies with empty evidence")
        for res in self.client.map(reask, label=f"{self.step} judge re-ask"):
            stem, tag = res.call.meta["stem"], res.call.meta["repeat"]
            ans = out[(stem, tag)]
            if not res.ok:
                self.failed.append((res.call.item_id, res.error))
                continue
            names = res.call.meta["codes"]
            for n, j in judge_answers(res.value, names).items():
                a = _answer(j)
                a["reasked"] = True
                if _unsupported(a):
                    a.update(present=False, unsupported=True, switchable_off=False)
                ans[n] = a
        return out


def judge_rules(client, cfg, step, codes, versions, families=None):
    """Judge every rule against every code with repeats and a tie-break.

    Returns (final rows, repeat rows, failed [(item, error)]). Raises StepFailed
    beyond the judge failure budget (default 0).
    """
    repeats = int(cfg.get("judge", {}).get("repeats", 2))
    judge = _Judge(client, cfg, step, codes, versions, families)
    orders = [code_order(codes, r, judge.seed) for r in range(repeats)]
    stems = [v["stem"] for v in versions]
    jobs = [
        (s, f"r{r}", orders[r], [c["name"] for c in orders[r]])
        for r in range(repeats)
        for s in stems
    ]
    answers = judge.run(jobs)
    # disagreements -> one tie-break run per rule, in a further order
    tb_order = code_order(codes, repeats, judge.seed)
    tb_jobs = []
    for s in stems:
        per = [answers.get((s, f"r{r}")) for r in range(repeats)]
        if any(p is None for p in per):
            continue
        split = {
            c["name"] for c in codes if len({p[c["name"]]["present"] for p in per}) > 1
        }
        if split:
            names = [c["name"] for c in tb_order if c["name"] in split]
            tb_jobs.append((s, TIEBREAK, tb_order, names))
    if tb_jobs:
        log(f"  {step}: tie-break for {len(tb_jobs)} of {len(stems)} rules")
    answers.update(judge.run(tb_jobs))
    failed_stems = sorted({i.split("#")[0] for i, _ in judge.failed})
    check_failures(
        cfg,
        step,
        "judge",
        [(s, "a judge call failed") for s in failed_stems],
        len(stems),
    )
    rows, repeat_rows = [], []
    for s in stems:
        if s in failed_stems:
            continue
        rule = judge.versions[s]["rule"]
        per = [answers[(s, f"r{r}")] for r in range(repeats)]
        tb = answers.get((s, TIEBREAK), {})
        for c in codes:
            n = c["name"]
            mine = [p[n] for p in per]
            for r, a in enumerate(mine):
                repeat_rows.append({"version": s, "what_code": n, "repeat": r, **a})
            if n in tb:
                repeat_rows.append(
                    {"version": s, "what_code": n, "repeat": TIEBREAK, **tb[n]}
                )
            a, used_tb = resolve(mine, tb.get(n))
            rows.append(
                {
                    "version": s,
                    "what_code": n,
                    "present": a["present"],
                    "switchable_off": a["switchable_off"] if a["present"] else None,
                    "quote": a["evidence"] if a["present"] else "",
                    "quote_ok": (
                        quote_in_source(a["evidence"], rule) if a["present"] else None
                    ),
                    # unsupported if any repeat (or the tie-break) was
                    "unsupported": any(m["unsupported"] for m in mine)
                    or bool(tb.get(n, {}).get("unsupported")),
                    "repeats": [m["present"] for m in mine],
                    "tiebreak": used_tb,
                }
            )
    return rows, repeat_rows, [(s, "a judge call failed") for s in failed_stems]


def reliability(repeat_rows):
    """Test-retest agreement of repeat 0 and repeat 1, per code and overall."""
    by = {}
    for r in repeat_rows:
        if r["repeat"] in (0, 1):
            by.setdefault((r["version"], r["what_code"]), {})[r["repeat"]] = r[
                "present"
            ]
    pairs = {}
    for (_, code), d in by.items():
        if 0 in d and 1 in d:
            pairs.setdefault(code, []).append((d[0], d[1]))
    out = [dict(code=c, **kappa_stats(p)) for c, p in sorted(pairs.items())]
    allp = [p for ps in pairs.values() for p in ps]
    out.append(dict(code="__overall__", **kappa_stats(allp)))
    return out


def write_reliability(path, rows):
    cols = [
        "code",
        "n",
        "both_yes",
        "both_no",
        "a_only",
        "b_only",
        "agreement",
        "kappa",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for row in rows:
        w.writerow(
            {
                k: ("" if v is None else round(v, 4) if isinstance(v, float) else v)
                for k, v in row.items()
            }
        )
    atomic_write_text(path, buf.getvalue())
