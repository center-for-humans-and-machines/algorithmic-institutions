#!/usr/bin/env bash
# Regenerate every summary output from the committed inputs.
#
# The outputs (report_bundle.html and its siblings) are deliberately not
# tracked -- see notes/autoresearch_summary.md §2. This is how you get
# them back: steps 3-8 of that pipeline, in order, from the committed
# data/*.json caches and the committed stack_visuals figures. Steps 1-2
# (collect, classify) are frozen with the corpus and are NOT run here.
#
# The last step prints the step-9 publish call.
#
# Usage:  scripts/data_analysis/autoresearch_summary/regenerate.sh
set -euo pipefail

cd "$(dirname "$0")/../../.."
DIR=scripts/data_analysis/autoresearch_summary
export PYTHONPATH="${DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PY=".venv/bin/python"
[ -x "${PY}" ] || PY="$(command -v python3)"

for step in score_progressions leaderboard stack_visuals machinery \
            build_report bundle_report; do
    echo "== ${step}"
    "${PY}" "${DIR}/${step}.py"
done
