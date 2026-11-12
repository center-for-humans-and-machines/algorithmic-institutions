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
      punishment = 5 * (c[..., -1] < c0)

`code` runs with these names in scope (rule_inputs computes them):
  the inputs of the rule's setting (SETTINGS), the raw history of every player
  (#245). For `exp2` (the default: two groups of 8 players in all, switching):
    c, valid, in_group   [B, A, t+1] float, rounds 0..t
    p                    [B, A, t] float, rounds 0..t-1
    t                    the round, 0-d float
  For `exp1` (one group of 4, no switching): c, valid, p and t, with a
  timed-out player's c read as 0.
  th        torch
  each name declared under `params`, from the params (0-d float tensor)
It must set `punishment`, one value per player ([B, A], or anything that
broadcasts to it); the manager then clamps it to [0, 30], casts to integers
and punishes only its own group. Python builtins are not available: use `th`
for the maths. The code is checked against the declaration, never used to
discover it.
"""

import ast
import inspect
import json
import math

import numpy as np
import torch as th
import yaml

#: What the rule code sees at the moment the manager punishes round t: after
#: round t's contributions, before any of its punishments. Every player of
#: both groups (axis 1, a player keeps its index for the game) over the rounds
#: so far (last axis), as the RL manager sees them (rule_inputs; the agent
#: prompt lists the same).
RULE_INPUTS = (
    "c",  # contribution in rounds 0..t, 0-20 (the timeout value if timed out)
    "valid",  # 1 where the player contributed, 0 where timed out, rounds 0..t
    "p",  # punishment charged in rounds 0..t-1, by either manager
    "in_group",  # 1 where the player was in this manager's group, rounds 0..t
    "t",  # the round, 0-23
)
#: The games a rule can be written for. `inputs`: the names its code sees;
#: `n_players` and `n_groups`: the layout validate_rule's random histories
#: take; `zero_timeouts`: whether rule_inputs reads a timed-out player's
#: contribution as 0 whatever the env recorded (experiment 1's clones are fed
#: their default, 15, which is no contribution).
SETTINGS = {
    "exp2": {
        "inputs": RULE_INPUTS,
        "n_players": 8,
        "n_groups": 2,
        "zero_timeouts": False,
    },
    "exp1": {
        "inputs": ("c", "valid", "p", "t"),
        "n_players": 4,
        "n_groups": 1,
        "zero_timeouts": True,
    },
}
#: exp2_multi: experiment 2's game and inputs; only the policy finder's score
#: differs (its sweeps play several governors, generate_sim_config.py).
SETTINGS["exp2_multi"] = dict(SETTINGS["exp2"])
#: exp2_diff: experiment 2's game and inputs, scored by the rule's group's lead
#: in common pool over `ah`'s group (generate_sim_config.py).
SETTINGS["exp2_diff"] = dict(SETTINGS["exp2"])
DEFAULT_SETTING = "exp2"
#: Names no param may take, in any setting.
ALL_INPUTS = tuple(dict.fromkeys(n for s in SETTINGS.values() for n in s["inputs"]))


def setting_of(name):
    """The SETTINGS entry `name` (None: the default)."""
    name = DEFAULT_SETTING if name is None else name
    if name not in SETTINGS:
        raise ValueError(f"unknown rule setting {name!r}, expected one of {sorted(SETTINGS)}")
    return SETTINGS[name]
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


#: torch submodules a rule may call into, as th.<submodule>.<function>; every
#: other submodule (th.os, th.builtins, th.nn, ...) is out of reach
TH_SUBMODULES = ("linalg", "special", "fft")
#: Attribute names a rule may not reach on th or on any value: file and pickle
#: I/O, global state of the sim (seeds, defaults, threads), compilation, and
#: string formatting (which reads attributes by name). With _DENIED_PREFIXES
#: and the ban on `_` names, this closes the ways from `th` or a value to
#: builtins, modules and frames. It raises the bar for rule code run on the
#: cluster; it is not a full sandbox.
_DENIED_ATTRIBUTES = {
    "load", "save", "from_file", "compile", "hub", "manual_seed", "seed",
    "initial_seed", "get_rng_state", "use_deterministic_algorithms",
    "format", "format_map", "mro",
}
#: set_* (global defaults, rng state, threads) and the frame, code and
#: traceback attributes of generators, coroutines and functions
_DENIED_PREFIXES = ("set_", "gi_", "cr_", "ag_", "f_", "co_", "tb_", "func_")
#: Statements and expressions a rule has no use for, which hand out functions,
#: frames or scopes
_DENIED_NODES = (
    ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
    ast.GeneratorExp, ast.Yield, ast.YieldFrom, ast.Await, ast.Global,
    ast.Nonlocal, ast.With, ast.AsyncWith, ast.AsyncFor, ast.Try, ast.Raise,
    ast.Delete,
)


def _check_attributes(tree, module=th):
    """Reject attribute access and constructs that leave the rule's maths.

    `th` may only be read as `th.<name>`: a torch function or constant, or
    `th.<submodule>.<function>` for TH_SUBMODULES. Any attribute (on th or on
    a value, e.g. a tensor method) must not start with `_`, be in
    _DENIED_ATTRIBUTES or start with _DENIED_PREFIXES.
    """
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in ast.walk(tree):
        if isinstance(node, _DENIED_NODES):
            raise ValueError(f"`{type(node).__name__}` is not allowed")
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if (
                attr.startswith("_")
                or attr in _DENIED_ATTRIBUTES
                or attr.startswith(_DENIED_PREFIXES)
            ):
                raise ValueError(f"attribute `{attr}` is not allowed")
        if isinstance(node, ast.Name) and node.id in RULE_MODULES:
            if not isinstance(node.ctx, ast.Load):
                raise ValueError(f"`{node.id}` cannot be assigned")
            parent = parents.get(node)
            if not (isinstance(parent, ast.Attribute) and parent.value is node):
                raise ValueError(f"`{node.id}` can only be used as {node.id}.<function>")
            value = getattr(module, parent.attr, None)
            if value is None:
                raise ValueError(f"{node.id}.{parent.attr} does not exist")
            if inspect.ismodule(value):
                if parent.attr not in TH_SUBMODULES:
                    raise ValueError(f"{node.id}.{parent.attr} is not allowed")
                grand = parents.get(parent)
                if not (isinstance(grand, ast.Attribute) and grand.value is parent):
                    raise ValueError(
                        f"{node.id}.{parent.attr} can only be used as"
                        f" {node.id}.{parent.attr}.<function>"
                    )
                inner = getattr(value, grand.attr, None)
                if inner is None or inspect.ismodule(inner):
                    raise ValueError(f"{node.id}.{parent.attr}.{grand.attr} is not allowed")


#: th functions that make a new tensor on the CPU unless given `device=`. The
#: sweep runs rules on the GPU (inputs and params on c.device), where such a
#: tensor fails every point, while CPU checks pass: rejected up front.
_CONSTRUCTORS = {
    "arange", "zeros", "ones", "full", "empty", "tensor", "as_tensor",
    "linspace", "logspace", "eye", "rand", "randn", "randint", "range",
}


def _check_devices(tree):
    """Reject th.<constructor>(...) without `device=` (see _CONSTRUCTORS)."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        func = node.func
        if (isinstance(func.value, ast.Name) and func.value.id == "th"
                and func.attr in _CONSTRUCTORS
                and not any(k.arg == "device" for k in node.keywords)):
            raise ValueError(
                f"th.{func.attr}(...) needs device=c.device: the sweep runs the rule"
                f" on the GPU, where a tensor made on the CPU cannot meet the inputs"
                f" (or use th.zeros_like, th.ones_like, th.full_like on an input)"
            )


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


def read_rule(rule_path, setting=None):
    """Read and check a rule YAML on its own; return (rule, compiled code).

    `setting` (SETTINGS, default exp2) sets the inputs the code may read.

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
    inputs = setting_of(setting)["inputs"]
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
        set(declared) & {*ALL_INPUTS, *RULE_MODULES, RULE_OUTPUT, PARAMS_BEST}
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
        tree = ast.parse(code, rule_path, "exec")
        read, bound = _names(tree)
        _check_attributes(tree)
        _check_devices(tree)
    except ValueError as e:
        raise ValueError(f"{rule_path}: code: {e}") from None
    undefined = sorted(
        read - bound - set(inputs) - set(RULE_MODULES) - set(declared)
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


def load_rule(rule_path, params, setting=None):
    """Read a rule YAML and its params; return (compiled code, params).

    `params` is a JSON path or a mapping (inline in a sim config). If it has
    a `best` key, the set under `best` is loaded and the rest ignored, so a
    sweep's `sweep.json` loads as params. See read_rule and check_params for
    the checks.
    """
    rule, code = read_rule(rule_path, setting)
    if isinstance(params, dict):
        source = "inline params"
    else:
        source = params
        with open(params) as f:
            params = json.load(f)
    if isinstance(params, dict) and PARAMS_BEST in params:
        params, source = params[PARAMS_BEST], f"{source} `{PARAMS_BEST}`"
    return code, check_params(rule_path, rule, params, source)


def rule_inputs(contribution, valid, punishment, agent_group, group, setting=None):
    """The setting's inputs from the round history: `contribution`, `valid`
    and `agent_group` (each player's group, 0 or 1) are [B, A, t+1] over
    episodes B, players A of both groups and rounds 0..t; `punishment` is
    [B, A, t], the punishments charged in rounds 0..t-1. `group` is the group
    the manager runs, which `in_group` is relative to. Returns float tensors;
    `t` is 0-d.
    """
    spec = setting_of(setting)
    contribution = contribution.to(th.float)
    if spec["zero_timeouts"]:
        contribution = th.where(valid.to(th.bool), contribution, th.zeros_like(contribution))
    if punishment.shape[-1] != contribution.shape[-1] - 1:
        raise ValueError(
            f"rule_inputs: punishment covers {punishment.shape[-1]} rounds,"
            f" contribution {contribution.shape[-1]}: expected one fewer"
        )
    inputs = {
        "c": contribution,
        "valid": valid.to(th.float),
        "p": punishment.to(th.float),
        "in_group": (agent_group == group).to(th.float),
        "t": th.tensor(
            contribution.shape[-1] - 1, dtype=th.float, device=contribution.device
        ),
    }
    return {k: inputs[k] for k in spec["inputs"]}


def run_rule(code, params, inputs):
    """Run compiled rule code on `inputs` (rule_inputs); return the raw float
    punishment, one per player [B, A] (before the manager's clamp and integer
    cast)."""
    # the inputs go in as copies, so in-place ops in the rule cannot touch
    # them; no builtins, so the maths goes through th. Params are 0-d float
    # tensors on the inputs' device, so th functions take them alone too
    # (th.log(p)), not only combined with an input
    c = inputs["c"]
    scope = {
        "__builtins__": {},
        "th": th,
        **{k: th.tensor(v, dtype=th.float, device=c.device) for k, v in params.items()},
        **{k: v.clone() for k, v in inputs.items()},
    }
    exec(code, scope)
    if RULE_OUTPUT not in scope:
        raise ValueError("rule_based: the rule code did not set `punishment`")
    # a scalar rule (e.g. a constant) is broadcast to every player; the
    # manager's cast floors non-integer values after the clamp
    raw = th.as_tensor(scope[RULE_OUTPUT], dtype=th.float, device=c.device)
    if th.isnan(raw).any():
        raise ValueError("rule_based: the rule code produced NaN punishments")
    players = c.shape[:-1]
    try:
        return raw.broadcast_to(players)
    except RuntimeError:
        raise ValueError(
            f"rule_based: `punishment` has shape {list(raw.shape)}, expected one"
            f" value per player {list(players)}"
        ) from None


#: The rounds validate_rule runs a rule at: the first (no punishment yet), a
#: middle one and the last.
DRY_RUN_ROUNDS = (0, 11, 23)


def dry_run_inputs(n_states=2048, n_players=None, rounds=DRY_RUN_ROUNDS, seed=0, setting=None):
    """rule_inputs over random histories, one set per round in `rounds`, for
    validate_rule: contributions 0-20 with timeouts, punishments 0-30, and,
    with two groups, each round's group split anywhere from all players in
    one group to an even split, and each state's manager running group 0 or
    1. The player count and groups come from the setting."""
    spec = setting_of(setting)
    n_players = spec["n_players"] if n_players is None else n_players
    g = th.Generator().manual_seed(seed)
    out = []
    for t in rounds:
        shape = (n_states, n_players, t + 1)
        split = th.rand(n_states, 1, t + 1, generator=g)  # per state and round
        if spec["n_groups"] == 1:
            split = th.zeros_like(split)  # every player in group 0
        out.append(
            rule_inputs(
                contribution=th.randint(0, 21, shape, generator=g),
                valid=th.rand(shape, generator=g) > 0.1,
                punishment=th.randint(0, 31, (n_states, n_players, t), generator=g),
                agent_group=(th.rand(shape, generator=g) < split).to(th.long),
                group=th.randint(0, spec["n_groups"], (n_states, 1, 1), generator=g),
                setting=setting,
            )
        )
    return out


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


def validate_rule(
    rule_path, min_params=1, max_params=4, n_points=SOBOL_POINTS, setting=None
):
    """Check a rule for a sweep; return its design (see sobol_design).

    On top of read_rule: `sweep_config` is required, the rule declares
    `min_params` to `max_params` params (None: no upper bound, #244), every
    design point passes
    check_params (constraints included), and the code runs on every design
    point over random histories at the first, a middle and the last round
    (dry_run_inputs) without error or NaN, giving one punishment per player.
    """
    rule, code = read_rule(rule_path, setting)
    if "sweep_config" not in rule:
        raise ValueError(f"{rule_path}: needs a `sweep_config` for a sweep")
    n_params = len(rule["params"])
    if n_params < min_params or (max_params is not None and n_params > max_params):
        allowed = (
            f"at least {min_params}"
            if max_params is None
            else f"{min_params} to {max_params}"
        )
        raise ValueError(f"{rule_path}: declares {n_params} params, allowed {allowed}")

    design = sobol_design(rule, n_points)
    histories = dry_run_inputs(setting=setting)
    for i, point in enumerate(design):
        source = f"{rule_path}: design point s{i:03d} {point}"
        check_params(rule_path, rule, point, source)
        for inputs in histories:
            try:
                run_rule(code, point, inputs)
            except Exception as e:
                raise ValueError(f"{source}, round {int(inputs['t'])}: {e}") from None
    return design
