#!/usr/bin/env python3
"""PreToolUse guard for a policy-finder instance (#236).

Registered by scripts/policy_finder/new_instance.sh in the instance's
.claude/settings.local.json, so it guards every call of the session (the
wrapper and the policy-finder subagent alike). Reads the hook input JSON on
stdin and prints a PreToolUse deny decision, or nothing to let the call
through to the permission rules and the sandbox.

- Write/Edit/NotebookEdit: only configs/managers/rule_based/<name>.yml,
  notes/policy_finder/<name>.md and scripts/policy_finder/<name>/**
- Read/Grep/Glob: only inside the worktree, never its .git
- Bash: no `git` or `gh` anywhere in the command
- WebFetch, WebSearch and MCP tools: never

<name> comes from .claude/policy_finder.json, written by the launcher. The
worktree root is two levels above this file. Anything the guard cannot
check (bad input, missing instance file) is denied.
"""

import json
import os
import re
import sys

ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
INSTANCE_FILE = os.path.join(ROOT, ".claude", "policy_finder.json")

WRITE_TOOLS = {"Write", "Edit", "NotebookEdit"}
READ_TOOLS = {"Read", "Grep", "Glob"}
DENIED_TOOLS = {"WebFetch", "WebSearch"}
# a git/gh command word: not part of a longer name (digit, github), and not
# hidden behind a path (/usr/bin/git), quotes ('git'), $( or backticks
GIT_RE = re.compile(r"(?:^|[\s;&|()`'\"=/<>{}])(?:git|gh)(?=$|[\s;&|()`'\"<>{}])")


class Denied(Exception):
    pass


def _resolve(path, cwd):
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(cwd, path)
    return os.path.realpath(path)


def _inside(path, base):
    return path == base or path.startswith(base + os.sep)


def write_paths():
    try:
        with open(INSTANCE_FILE) as f:
            name = json.load(f)["name"]
    except (OSError, ValueError, KeyError, TypeError):
        raise Denied(f"no instance file at {INSTANCE_FILE}: writes are disabled")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise Denied(f"bad instance name {name!r} in {INSTANCE_FILE}")
    rule = os.path.join(ROOT, "configs", "managers", "rule_based", f"{name}.yml")
    notes = os.path.join(ROOT, "notes", "policy_finder", f"{name}.md")
    scripts = os.path.join(ROOT, "scripts", "policy_finder", name)
    return rule, notes, scripts


def check_write(tool_input, cwd):
    path = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not isinstance(path, str) or not path:
        raise Denied("no file path to check")
    path = _resolve(path, cwd)
    rule, notes, scripts = write_paths()
    if path not in (rule, notes) and not _inside(path, scripts):
        raise Denied(
            f"writes are limited to {os.path.relpath(rule, ROOT)}, "
            f"{os.path.relpath(notes, ROOT)} and {os.path.relpath(scripts, ROOT)}/"
        )


def check_read(tool_name, tool_input, cwd):
    if tool_name == "Read":
        paths = [tool_input.get("file_path")]
        if not isinstance(paths[0], str) or not paths[0]:
            raise Denied("no file path to check")
    else:
        paths = [tool_input.get("path") or cwd]
    if tool_name == "Glob":
        pattern = tool_input.get("pattern", "")
        if ".." in pattern.split("/"):
            raise Denied("glob patterns may not climb with ..")
        if pattern.startswith(("/", "~")):
            # the literal prefix before the first wildcard is where it searches
            paths.append(re.split(r"[*?\[{]", pattern, maxsplit=1)[0] or "/")
    git_dir = os.path.join(ROOT, ".git")
    for path in paths:
        path = _resolve(path, cwd)
        if not _inside(path, ROOT):
            raise Denied("reads are limited to the worktree")
        if _inside(path, git_dir):
            raise Denied(".git is not readable")


def check_bash(tool_input):
    command = tool_input.get("command")
    if not isinstance(command, str):
        raise Denied("no command to check")
    if GIT_RE.search(command):
        raise Denied("git and gh are not available")


def check(data):
    tool_name = data.get("tool_name")
    tool_input = data.get("tool_input") or {}
    cwd = data.get("cwd") or ROOT
    if not isinstance(tool_name, str) or not isinstance(tool_input, dict):
        raise Denied("malformed hook input")
    if tool_name in DENIED_TOOLS or tool_name.startswith("mcp__"):
        raise Denied(f"{tool_name} is not available")
    if tool_name in WRITE_TOOLS:
        check_write(tool_input, cwd)
    elif tool_name in READ_TOOLS:
        check_read(tool_name, tool_input, cwd)
    elif tool_name == "Bash":
        check_bash(tool_input)


def main():
    try:
        check(json.load(sys.stdin))
    except Denied as e:
        reason = str(e)
    except Exception as e:  # fail closed on anything unexpected
        reason = f"guard error: {e!r}"
    else:
        return
    decision = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"policy-finder guard: {reason}",
        }
    }
    print(json.dumps(decision))


if __name__ == "__main__":
    main()
