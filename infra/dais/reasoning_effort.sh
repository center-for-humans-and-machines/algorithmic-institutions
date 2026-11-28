# resolve_reasoning_effort: a submission's REASONING_EFFORT (or the profile's
# REASONING_EFFORT_DEFAULT when it sets none), sourced by common.sh after the
# model profile. A profile that has an effort declares:
#   REASONING_EFFORT_DEFAULT  the effort a job runs at unless it sets one
#   REASONING_EFFORT_LEVELS   "name=budget ..." the named efforts and what the
#                             prompt encoder renders for each
#   REASONING_EFFORT_RANGE    "min max": integers the model also takes (empty: none)
#   REASONING_EFFORT_PROMPT   the rendered prompt's text for a budget ({budget}),
#                             checked on the running server (check_prompt)
# It appends the server's default to MODEL_VLLM_ARGS
# (--default-chat-template-kwargs {"reasoning_effort":...}) and sets
# PROMPT_MUST_CONTAIN, REASONING_EFFORT (the effort in force) and
# REASONING_EFFORT_BUDGET. A model without an effort (no levels) refuses a
# REASONING_EFFORT; an invalid one fails before the server starts.
resolve_reasoning_effort() {
  local effort="${REASONING_EFFORT:-${REASONING_EFFORT_DEFAULT:-}}" budget="" value
  if [ -z "${REASONING_EFFORT_LEVELS:-}" ]; then
    if [ -n "${REASONING_EFFORT:-}" ]; then
      echo "FATAL: REASONING_EFFORT=$REASONING_EFFORT, but model ${MODEL:-?} has no" \
        "reasoning effort (its profile declares none)" >&2
      return 1
    fi
    REASONING_EFFORT="" REASONING_EFFORT_BUDGET=""
    return 0
  fi
  for level in $REASONING_EFFORT_LEVELS; do
    [ "${level%%=*}" = "$effort" ] && budget="${level#*=}"
  done
  if [ -n "$budget" ]; then
    value="\"$effort\""
  elif [[ "$effort" =~ ^[0-9]+$ ]] && [ -n "${REASONING_EFFORT_RANGE:-}" ] \
      && (( 10#$effort >= ${REASONING_EFFORT_RANGE% *} \
            && 10#$effort <= ${REASONING_EFFORT_RANGE#* } )); then
    budget=$(( 10#$effort )) effort=$(( 10#$effort )) value=$(( 10#$effort ))
  else
    echo "FATAL: reasoning effort '$effort' is not one of ${MODEL:-the model}'s:" \
      "${REASONING_EFFORT_LEVELS}${REASONING_EFFORT_RANGE:+, or an integer" \
      "${REASONING_EFFORT_RANGE% *} to ${REASONING_EFFORT_RANGE#* }}" >&2
    return 1
  fi
  REASONING_EFFORT="$effort" REASONING_EFFORT_BUDGET="$budget"
  MODEL_VLLM_ARGS="${MODEL_VLLM_ARGS:+$MODEL_VLLM_ARGS }--default-chat-template-kwargs"
  MODEL_VLLM_ARGS+=" {\"reasoning_effort\":$value}"
  PROMPT_MUST_CONTAIN="${REASONING_EFFORT_PROMPT//\{budget\}/$budget}"
}
