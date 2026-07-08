#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="${1:?run dir}"
MAIN_PID="${2:?main pid}"
OUT="$RUN_DIR/system_monitor.jsonl"
echo "# writing monitor JSONL to $OUT"
while true; do
  TS="$(date -Is)"
  MAIN_UPTIME=""
  if ps -p "$MAIN_PID" >/dev/null 2>&1; then
    MAIN_UPTIME="$(ps -p "$MAIN_PID" -o etimes= | tr -d ' ')"
  fi
  LOG_BYTES=0
  [[ -f "$RUN_DIR/train.log" ]] && LOG_BYTES="$(stat -c %s "$RUN_DIR/train.log")"
  EVENT_LINES=0
  [[ -f "$RUN_DIR/events.jsonl" ]] && EVENT_LINES="$(wc -l < "$RUN_DIR/events.jsonl")"
  VIDEO_LINES=0
  [[ -f "$RUN_DIR/video_manifest.jsonl" ]] && VIDEO_LINES="$(wc -l < "$RUN_DIR/video_manifest.jsonl")"
  ROCM="$(rocm-smi --showuse --showmemuse --showpidgpus 2>/dev/null | tr '\n' ' ' | sed 's/"/\\"/g')"
  printf '{"ts":"%s","main_pid":%s,"main_uptime_s":"%s","train_log_bytes":%s,"event_lines":%s,"video_lines":%s,"rocm_smi":"%s"}\n' \
    "$TS" "$MAIN_PID" "$MAIN_UPTIME" "$LOG_BYTES" "$EVENT_LINES" "$VIDEO_LINES" "$ROCM" >> "$OUT"
  if [[ -z "$MAIN_UPTIME" ]]; then
    printf '{"ts":"%s","event":"process_exit_observed","main_pid":%s}\n' "$TS" "$MAIN_PID" >> "$OUT"
    exit 0
  fi
  sleep 60
done
