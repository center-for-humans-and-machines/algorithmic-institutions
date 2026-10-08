"""Config-defined rules for RuleBasedManager: load, run, design a sweep, validate.

Kept free of PyG imports, so `python -m aimanager validate-rule` and its tests
run locally; `api_manager.RuleBasedManager` builds on it.

A rule YAML:

    params:                # every fittable param
      c0:
        definition: contribution at which punishment starts
        type: int          # int or float
    sweep_config:          # per param: a number (fixed), [low, high] or
      c0: [0, 20]          # [low, high, log] (log-uniform, low > 0); an int
                           # range takes each integer low..high, equally often
    constraints:           # optional comparisons over params only
      - c0 >= 0
    code: |
      punishment = 5 * (c < c0)

`code` runs with these names in scope (rule_inputs computes them):
  RULE_INPUTS, each a float tensor with one value per player:
    the player:  c, valid, c_prev, p_prev, t
    own group:   n, c_group
    other group: n_other, c_other, p_prev_other
  th        torch
  each name declared under `params`, from the params (0-d float tensor)
It must set `punishment`; the manager then clamps it to [0, 30] and casts to
integers. Python builtins are not available: use `th` for the maths. The code
is checked against the declaration, never used to discover it.
"""

import ast
import json
import math

import numpy as np
import torch as th
import yaml

#: What the rule code sees, one value per player, at the moment the manager
#: punishes round t: after round t's contributions, before any of its
#: punishments (rule_inputs; the agent prompt lists the same).
RULE_INPUTS = (
    "c",  # own contribution this round, 0-20 (the timeout value if timed out)
    "valid",  # 1 if the player contributed this round, 0 if timed out
    "c_prev",  # own contribution last round (round 0: c)
    "p_prev",  # own punishment last round (round 0: 0)
    "t",  # round number, 0-23
    "n",  # own group's size, the player included
    "c_group",  # mean contribution of the rest of the own group, valid only
    "n_other",  # the other group's size
    "c_other",  # the other group's mean contribution, valid only
    "p_prev_other",  # last round's mean punishment of the players now in the
    # other group
)
#: Besides the inputs and the params.
RULE_MODULES = ("th",)
RULE_OUTPUT = "punishment"
RULE_KEYS = {"params", "sweep_config", "constraints", "code"}
#: The fields every declared param carries, and the types it may have.
PARAM_FIELDS = {"definition", "type"}
PARAM_TYPES = ("int", "float")
#: The marker that makes a `sweep_config` range log-uniform.
LOG_SCALE = "log"
#: The key under which a params JSON (a sweep's `sweep.json`) holds the set to
#: load; everything else in such a file is the sweep's record.
PARAMS_BEST = "best"


def _names(tree):
    """(names read, names bound) anywhere in a parsed snippet."""
    read, bound = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed")
        if isinstance(node, ast.Name):
            (read if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    return read, bound


def _is_number(value):
    """A finite int or float; bools are not numbers here."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    return is_number and math.isfinite(value)


def _check_declared(rule_path, declared):
    """Each declared param is a mapping of exactly `definition` and `type`."""
    for name, spec in declared.items():
        if not isinstance(spec, dict) or set(spec) != PARAM_FIELDS:
            raise ValueError(
                f"{rule_path}: param `{name}` must declare exactly"
                f" {sorted(PARAM_FIELDS)}"
            )
        if not isinstance(spec["definition"], str) or not spec["definition"].strip():
            raise ValueError(f"{rule_path}: param `{name}` needs a `definition`")
        if spec["type"] not in PARAM_TYPES:
            raise ValueError(
                f"{rule_path}: param `{name}` has type {spec['type']!r},"
                f" not one of {list(PARAM_TYPES)}"
            )


def _check_sweep_config(rule_path, declared, sweep):
    """`sweep_config` gives every declared param a fixed value or a range."""
    if not isinstance(sweep, dict):
        raise ValueError(f"{rule_path}: `sweep_config` must be a mapping")
    missing = sorted(set(declared) - set(sweep))
    undeclared = sorted(set(sweep) - set(declared))
    if missing or undeclared:
        raise ValueError(
            f"{rule_path}: `sweep_config` does not match the declared params:"
            f" missing {missing}, undeclared {undeclared}"
        )
    for name, spec in sweep.items():
        where = f"{rule_path}: sweep_config `{name}`"
        if _is_number(spec):
            if declared[name]["type"] == "int" and spec != int(spec):
                raise ValueError(f"{where} = {spec} is not an integer")
            continue
        is_range = isinstance(spec, list) and len(spec) in (2, 3)
        if not is_range or not all(_is_number(v) for v in spec[:2]):
            raise ValueError(
                f"{where} must be a number, [low, high] or [low, high, log],"
                f" got {spec!r}"
            )
        low, high = spec[:2]
        if low >= high:
            raise ValueError(f"{where}: low {low} is not below high {high}")
        if declared[name]["type"] == "int" and (low != int(low) or high != int(high)):
            raise ValueError(
                f"{where}: an int range takes integer bounds, got [{low}, {high}]"
            )
        if len(spec) == 3:
            if spec[2] != LOG_SCALE:
                raise ValueError(f"{where}: third entry must be `log`, got {spec[2]!r}")
            if low <= 0:
                raise ValueError(f"{where}: a log range needs low > 0, got {low}")


def read_rule(rule_path):
    """Read and check a rule YAML on its own; return (rule, compiled code).

    Checked here:
      - the only top-level keys are `params`, `sweep_config`, `constraints`
        and `code`
      - `params` declares every fittable param with exactly a non-empty
        `definition` and a `type`; none may take the name of an input
        (RULE_INPUTS), th, punishment or `best`
      - `sweep_config` (optional here; `validate_rule` requires it) covers
        exactly the declared params; a fixed `int` is an integer, a range has
        low < high, and an `int` range has integer bounds
      - `code` reads only the inputs, th, declared params and names it assigns, reads
        every declared param, and assigns `punishment`
      - `constraints` (optional) are comparisons over declared params only
    """
    with open(rule_path) as f:
        rule = yaml.safe_load(f)

    if not isinstance(rule, dict):
        raise ValueError(f"{rule_path}: must be a YAML mapping")
    unknown = sorted(set(rule) - RULE_KEYS)
    if unknown:
        raise ValueError(f"{rule_path}: unknown keys {unknown}")

    declared = rule.get("params")
    if not isinstance(declared, dict) or not declared:
        raise ValueError(f"{rule_path}: needs a non-empty `params` mapping")
    reserved = sorted(
        set(declared) & {*RULE_INPUTS, *RULE_MODULES, RULE_OUTPUT, PARAMS_BEST}
    )
    if reserved:
        raise ValueError(f"{rule_path}: params use reserved names {reserved}")
    _check_declared(rule_path, declared)
    if "sweep_config" in rule:
        _check_sweep_config(rule_path, declared, rule["sweep_config"])

    code = rule.get("code")
    if not isinstance(code, str):
        raise ValueError(f"{rule_path}: needs a `code` block")
    try:
        read, bound = _names(ast.parse(code, rule_path, "exec"))
    except ValueError as e:
        raise ValueError(f"{rule_path}: code: {e}") from None
    undefined = sorted(
        read - bound - set(RULE_INPUTS) - set(RULE_MODULES) - set(declared)
    )
    if undefined:
        raise ValueError(f"{rule_path}: code reads undefined names {undefined}")
    unused = sorted(set(declared) - read)
    if unused:
        raise ValueError(f"{rule_path}: code never reads params {unused}")
    if RULE_OUTPUT not in bound:
        raise ValueError(f"{rule_path}: code never assigns `{RULE_OUTPUT}`")

    constraints = rule.get("constraints", [])
    if not isinstance(constraints, list):
        raise ValueError(f"{rule_path}: `constraints` must be a list")
    for expr in constraints:
        if not isinstance(expr, str):
            raise ValueError(f"{rule_path}: constraint {expr!r} is not a string")
        tree = ast.parse(expr, rule_path, "eval")
        if not isinstance(tree.body, ast.Compare):
            raise ValueError(f"{rule_path}: constraint `{expr}` is not a comparison")
        read, _ = _names(tree)
        undefined = sorted(read - set(declared))
        if undefined:
            raise ValueError(
                f"{rule_path}: constraint `{expr}` reads undeclared names {undefined}"
            )

    return rule, compile(code, rule_path, "exec")


def check_params(rule_path, rule, params, source):
    """Check a set of params against a read rule; return them as a new dict.

    The params hold exactly the declared names, as finite numbers, integers
    for `int` params, and satisfy every constraint.
    """
    declared = rule["params"]
    if not isinstance(params, dict):
        raise ValueError(f"{source}: must be a JSON object")
    params = dict(params)
    missing = sorted(set(declared) - set(params))
    undeclared = sorted(set(params) - set(declared))
    if missing or undeclared:
        raise ValueError(
            f"{source} does not match the params declared in {rule_path}:"
            f" missing {missing}, undeclared {undeclared}"
        )
    for name, value in params.items():
        if not _is_number(value):
            raise ValueError(f"{source}: {name} = {value!r} is not a finite number")
        if declared[name]["type"] == "int" and value != int(value):
            raise ValueError(f"{source}: {name} = {value} is not an integer")

    for expr in rule.get("constraints", []):
        tree = ast.parse(expr, rule_path, "eval")
        if not eval(compile(tree, rule_path, "eval"), {"__builtins__": {}}, params):
            read, _ = _names(tree)
            values = ", ".join(f"{n} = {params[n]}" for n in sorted(read))
            raise ValueError(f"{source}: constraint `{expr}` failed: {values}")
    return params


def load_rule(rule_path, params):
    """Read a rule YAML and its params; return (compiled code, params).

    `params` is a JSON path or a mapping (inline in a sim config). If it has
    a `best` key, the set under `best` is loaded and the rest ignored, so a
    sweep's `sweep.json` loads as params. See read_rule and check_params for
    the checks.
    """
    rule, code = read_rule(rule_path)
    if isinstance(params, dict):
        source = "inline params"
    else:
        source = params
        with open(params) as f:
            params = json.load(f)
    if isinstance(params, dict) and PARAMS_BEST in params:
        params, source = params[PARAMS_BEST], f"{source} `{PARAMS_BEST}`"
    return code, check_params(rule_path, rule, params, source)


def _mean(total, count):
    """total / count, 0 where count is 0 (an empty group, no valid player)."""
    return th.where(count > 0, total / count.clamp(min=1), th.zeros_like(total))


def rule_inputs(
    contribution, valid, prev_contribution, prev_punishment, agent_group, round_number
):
    """RULE_INPUTS from the env state's tensors, all [B, A, 1] over episodes B
    and players A of both groups; `agent_group` holds each player's group
    (0 or 1). Returns float tensors of the same shape.

    Group means count only players who contributed this round (`valid`);
    `c_group` leaves the player out. A mean over nobody is 0: `n` and
    `n_other` tell the empty cases apart. In round 0, which has no last round,
    `c_prev` is `c` and `p_prev` is 0.
    """
    c = contribution.to(th.float)
    t = round_number.to(th.float)
    v = valid.to(th.float)
    first = t == 0
    c_prev = th.where(first, c, prev_contribution.to(th.float))
    p_prev = th.where(first, th.zeros_like(c), prev_punishment.to(th.float))

    group = agent_group.to(th.long)  # [B, A, 1]
    # [B, 2, A, 1]: 1 where player a is in group k
    member = th.stack([(group == k).to(th.float) for k in (0, 1)], 1)

    def per_group(x):  # [B, A, 1] -> [B, 2, 1] sums over each group
        return (member * x.unsqueeze(1)).sum(2)

    def own(x):  # [B, 2, 1] -> [B, A, 1], each player's own group's value
        return x.gather(1, group)

    def other(x):
        return x.gather(1, 1 - group)

    size, n_valid = per_group(th.ones_like(c)), per_group(v)
    c_sum, p_sum = per_group(v * c), per_group(p_prev)
    return {
        "c": c,
        "valid": v,
        "c_prev": c_prev,
        "p_prev": p_prev,
        "t": t,
        "n": own(size),
        "c_group": _mean(own(c_sum) - v * c, own(n_valid) - v),
        "n_other": other(size),
        "c_other": _mean(other(c_sum), other(n_valid)),
        "p_prev_other": _mean(other(p_sum), other(size)),
    }


def run_rule(code, params, inputs):
    """Run compiled rule code on `inputs` (rule_inputs); return the raw float
    punishment, broadcast to the inputs' shape (before the manager's clamp and
    integer cast)."""
    # the inputs go in as copies, so in-place ops in the rule cannot touch
    # them; no builtins, so the maths goes through th. Params are 0-d float
    # tensors on the inputs' device, so th functions take them alone too
    # (th.log(p)), not only combined with an input
    c = inputs["c"]
    scope = {
        "__builtins__": {},
        "th": th,
        **{k: th.tensor(v, dtype=th.float, device=c.device) for k, v in params.items()},
        **{k: inputs[k].clone() for k in RULE_INPUTS},
    }
    exec(code, scope)
    if RULE_OUTPUT not in scope:
        raise ValueError("rule_based: the rule code did not set `punishment`")
    # a scalar rule (e.g. a constant) is broadcast to every player and round;
    # the manager's cast floors non-integer values after the clamp
    raw = th.as_tensor(scope[RULE_OUTPUT], dtype=th.float, device=c.device)
    if th.isnan(raw).any():
        raise ValueError("rule_based: the rule code produced NaN punishments")
    return raw.broadcast_to(c.shape)


def dry_run_inputs(n_states=2048, n_players=8, seed=0):
    """rule_inputs over random env states, for validate_rule: every round,
    contributions 0-20 with timeouts, punishments 0-30, and group splits from
    all players in one group to an even split."""
    g = th.Generator().manual_seed(seed)
    shape = (n_states, n_players, 1)
    in_group_1 = th.rand(n_states, 1, 1, generator=g)  # each state its own split
    return rule_inputs(
        contribution=th.randint(0, 21, shape, generator=g),
        valid=th.rand(shape, generator=g) > 0.1,
        prev_contribution=th.randint(0, 21, shape, generator=g),
        prev_punishment=th.randint(0, 31, shape, generator=g),
        agent_group=(th.rand(shape, generator=g) < in_group_1).to(th.long),
        round_number=th.randint(0, 24, (n_states, 1, 1), generator=g).expand(shape),
    )


#: Sweep design defaults: Sobol points (a power of two keeps the design
#: balanced) and the scramble seed, fixed so a rule always gets one design.
SOBOL_POINTS = 256
SOBOL_SEED = 0


def sobol_design(rule, n_points=SOBOL_POINTS):
    """The sweep's design points, in order: a list of param dicts.

    A scrambled Sobol sequence over the ranged params of `sweep_config`
    (uniform, or uniform in log for `log` ranges), fixed params held at their
    value. An `int` range [low, high] is drawn over [low - 0.5, high + 0.5]
    and rounded to the nearest integer, so each integer low..high gets the same
    share (in log for `log` ranges); points that become identical are merged,
    so the design can hold fewer than `n_points`.
    """
    if n_points < 1 or n_points & (n_points - 1):
        raise ValueError(f"sobol points must be a power of two, got {n_points}")
    from scipy.stats import qmc

    declared, sweep = rule["params"], rule["sweep_config"]
    ranged = [n for n, s in sweep.items() if isinstance(s, list)]
    if ranged:
        sampler = qmc.Sobol(d=len(ranged), scramble=True, seed=SOBOL_SEED)
        u = sampler.random_base2(int(math.log2(n_points)))
    else:
        u = np.zeros((1, 0))

    design, seen = [], set()
    for row in u:
        point = {}
        for name, spec in sweep.items():
            is_int = declared[name]["type"] == "int"
            if name in ranged:
                low, high = spec[:2]
                if is_int:  # every integer gets a full unit of the range
                    low, high = low - 0.5, high + 0.5
                x = row[ranged.index(name)]
                if len(spec) == 3:
                    value = math.exp(math.log(low) + x * math.log(high / low))
                else:
                    value = low + x * (high - low)
            else:
                value = spec
            if is_int:
                value = int(math.floor(value + 0.5))
                if name in ranged:  # float error at the widened edges
                    value = min(max(value, int(spec[0])), int(spec[1]))
            else:
                value = float(value)
            point[name] = value
        key = tuple(point.items())
        if key not in seen:
            seen.add(key)
            design.append(point)
    return design


def validate_rule(rule_path, min_params=1, max_params=4, n_points=SOBOL_POINTS):
    """Check a rule for a sweep; return its design (see sobol_design).

    On top of read_rule: `sweep_config` is required, the rule declares
    `min_params` to `max_params` params, every design point passes
    check_params (constraints included), and the code runs on every design
    point over random env states (dry_run_inputs) without error or NaN.
    """
    rule, code = read_rule(rule_path)
    if "sweep_config" not in rule:
        raise ValueError(f"{rule_path}: needs a `sweep_config` for a sweep")
    n_params = len(rule["params"])
    if not min_params <= n_params <= max_params:
        raise ValueError(
            f"{rule_path}: declares {n_params} params, allowed {min_params}"
            f" to {max_params}"
        )

    design = sobol_design(rule, n_points)
    inputs = dry_run_inputs()
    for i, point in enumerate(design):
        source = f"{rule_path}: design point s{i:03d} {point}"
        check_params(rule_path, rule, point, source)
        try:
            run_rule(code, point, inputs)
        except Exception as e:
            raise ValueError(f"{source}: {e}") from None
    return design
