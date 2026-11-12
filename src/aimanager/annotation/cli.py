"""CLI: python -m aimanager.annotation <command> --run <run_id> [options]."""

import argparse
import os
import sys

from .common import (
    Run,
    check_revision_chain,
    file_hash,
    load_config,
    log,
    read_yaml,
    write_yaml,
)

LLM_COMMANDS = [
    "open-rules",
    "open-notes",
    "what-codebook",
    "coverage",
    "consolidate-what",
    "why-codebook",
    "consolidate-why",
    "basis-codebook",
    "consolidate-basis",
    "apply-rules",
    "apply-notes",
]


def _parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run", required=True, help="run id")
    common.add_argument("--config", help="YAML merged over the default config")
    common.add_argument(
        "--runs-root",
        default=os.environ.get("ANNOT_RUNS_ROOT", "runs/annotation"),
    )
    common.add_argument("--limit", type=int, help="first N items only")
    llm = argparse.ArgumentParser(add_help=False)
    llm.add_argument("--backend", choices=["http", "fake"], default="http")
    llm.add_argument("--base-url", default=os.environ.get("ANNOT_BASE_URL"))
    llm.add_argument("--model", default=os.environ.get("ANNOT_MODEL"))
    llm.add_argument("--concurrency", type=int, help="overrides llm.concurrency")

    p = argparse.ArgumentParser(prog="python -m aimanager.annotation")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("corpus", parents=[common], help="step 0")
    c.add_argument("--repo", help="git repository with the refs")
    for name in LLM_COMMANDS:
        sub.add_parser(name, parents=[common, llm])
    sub.add_parser("preflight", parents=[common, llm], help="one request per schema")
    s = sub.add_parser("stage", parents=[common, llm], help="A: 1a-3, B: 5a-5b")
    s.add_argument("which", choices=["A", "B"])
    e = sub.add_parser("expert-check", parents=[common])
    e.add_argument("action", choices=["export", "ingest"])
    e.add_argument("--sheets", help="directory of filled sheets (ingest)")
    r = sub.add_parser("review", parents=[common])
    r.add_argument("action", choices=["status", "begin"])
    r.add_argument("--kind", choices=["what", "why", "basis"], help="codebook (begin)")
    sub.add_parser("join", parents=[common])
    sub.add_parser("analyse", parents=[common], help="7: tables, figures, report")
    return p


def review_status(run):
    for kind in ("what", "why", "basis"):
        revs = check_revision_chain(run, kind)
        rev, path = revs[-1]
        others = ", ".join(p.name for _, p in revs[:-1]) or "none"
        print(f"{kind}: current {path.name} (older: {others})")
        print(
            f"  next revision for a reviewer: codebook/{kind}.v{rev + 1}.yml with "
            f"based_on: {{revision: {rev}, sha256: {file_hash(path)}}}"
        )
    comments = run.dir / "review" / "comments.md"
    print(f"review comments: {'present' if comments.exists() else 'none'}")


def review_begin(run, kind):
    """Copy the latest revision to the next one, with `based_on` filled in."""
    rev, path = check_revision_chain(run, kind)[-1]
    doc = read_yaml(path)
    nxt = path.with_name(f"{kind}.v{rev + 1}.yml")
    doc = {
        "kind": kind,
        "revision": rev + 1,
        "based_on": {"revision": rev, "sha256": file_hash(path)},
        **{k: v for k, v in doc.items() if k not in ("kind", "revision", "based_on")},
    }
    write_yaml(nxt, doc)
    print(f"wrote {nxt}; edit it in place")


def main(argv=None):
    args = _parser().parse_args(argv)
    cfg = load_config(args.config)
    run = Run(args.runs_root, args.run)
    if getattr(args, "concurrency", None):
        cfg["llm"]["concurrency"] = args.concurrency
    if args.command == "corpus":
        from .corpus import run_corpus

        run_corpus(run, cfg, limit=args.limit, repo=args.repo)
    elif args.command == "preflight":
        from .client import LLMClient
        from .preflight import preflight

        client = LLMClient(
            run,
            cfg,
            backend=args.backend,
            base_url=args.base_url,
            model=args.model,
            use_cache=False,
        )
        return 1 if preflight(client) else 0
    elif args.command in LLM_COMMANDS or args.command == "stage":
        from .client import LLMClient
        from .steps import STAGE_A, STAGE_B, STEPS

        client = LLMClient(
            run, cfg, backend=args.backend, base_url=args.base_url, model=args.model
        )
        if args.command == "stage":
            todo = STAGE_A if args.which == "A" else STAGE_B
        else:
            todo = [(args.command, STEPS[args.command])]
        for name, fn in todo:
            log(f"== {name}")
            fn(run, cfg, client, limit=args.limit)
    elif args.command == "expert-check":
        from . import expert

        if args.action == "export":
            expert.export(run, cfg, limit=args.limit)
        else:
            expert.ingest(run, cfg, sheets_dir=args.sheets)
    elif args.command == "review":
        if args.action == "begin":
            if not args.kind:
                raise SystemExit("review begin needs --kind what|why|basis")
            review_begin(run, args.kind)
        else:
            review_status(run)
    elif args.command == "join":
        from .join import join

        join(run, cfg, limit=args.limit)
    elif args.command == "analyse":
        from .analyse import analyse

        analyse(run, cfg, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
