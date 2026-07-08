#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="${1:?run dir}"
MAIN_PID="${2:?main pid}"
OUT="$RUN_DIR/system_monitor.jsonl"
echo "# writing monitor JSONL to $OUT"
while true; do
  TS="$(date -Is)"
  MAIN_ALIVE=0
  if kill -0 "$MAIN_PID" 2>/dev/null; then MAIN_ALIVE=1; fi
  LAST_EVENT=""
  if [ -s "$RUN_DIR/events.jsonl" ]; then
    LAST_EVENT=$(tail -1 "$RUN_DIR/events.jsonl" | python3 -c 'import json,sys; s=sys.stdin.read().strip(); print(json.dumps(json.loads(s)) if s else "")' 2>/dev/null || true)
  fi
  ROCM=$(rocm-smi --showuse --showmemuse --showpidgpus 2>/dev/null | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
  PS=$(ps -p "$MAIN_PID" -o pid=,ppid=,stat=,etimes=,pcpu=,pmem=,cmd= 2>/dev/null | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')
  printf '{"ts":%s,"main_pid":%s,"main_alive":%s,"last_event":%s,"ps":%s,"rocm_smi":%s}\n' \
    "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$TS")" \
    "$MAIN_PID" "$MAIN_ALIVE" "${LAST_EVENT:-null}" "$PS" "$ROCM" >> "$OUT"
  if [ "$MAIN_ALIVE" -eq 0 ]; then
    break
  fi
  sleep 60
done
