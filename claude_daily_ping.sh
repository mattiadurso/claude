#!/usr/bin/env bash
#
# claude_daily_ping.sh — fire one tiny Claude Code prompt every day at 07:30
# so the 5-hour usage session opens at 07:30 and resets at 12:30.
#
# The ping uses Haiku with no tools, no MCP, no skills and low effort:
# ~158 input + ~178 output tokens (~$0.001) per day.
#
#   ./claude_daily_ping.sh start        launch the daemon in the background
#   ./claude_daily_ping.sh stop         stop it
#   ./claude_daily_ping.sh status       is it running / when is the next ping
#   ./claude_daily_ping.sh now          send a ping right now (test)
#   ./claude_daily_ping.sh log          tail the log
#   ./claude_daily_ping.sh autostart    also start it automatically at login
#   ./claude_daily_ping.sh unautostart  undo autostart
#
set -uo pipefail

# ----------------------------------------------------------------- settings --
PING_HOUR="${PING_HOUR:-07}"       # hour of the daily ping (local time)
PING_MINUTE="${PING_MINUTE:-30}"   # minute of the daily ping
CATCHUP_MINUTES="${CATCHUP_MINUTES:-60}"  # if the machine was asleep at 08:00,
                                          # still ping if we wake up within this
                                          # many minutes (0 = never catch up)
MODEL="${MODEL:-claude-haiku-4-5-20251001}"
CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"

STATE_DIR="${STATE_DIR:-$HOME/.local/state/claude-daily-ping}"
LOG_FILE="$STATE_DIR/ping.log"
PID_FILE="$STATE_DIR/daemon.pid"
LAST_FILE="$STATE_DIR/last-ping-date"   # YYYY-MM-DD of the last successful ping

SELF="$(readlink -f "${BASH_SOURCE[0]}")"

mkdir -p "$STATE_DIR"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$LOG_FILE"; }

# --------------------------------------------------------------- the ping ----
# One non-interactive turn, everything optional switched off to keep the
# prompt (and therefore the token cost) as small as possible.
send_ping() {
  local attempt out usage result
  for attempt in 1 2 3; do
    out="$("$CLAUDE_BIN" -p "ping" \
            --model "$MODEL" \
            --effort low \
            --system-prompt "Reply with exactly: ok" \
            --tools "" \
            --strict-mcp-config \
            --mcp-config '{"mcpServers":{}}' \
            --disable-slash-commands \
            --no-session-persistence \
            --output-format json 2>&1)"
    if [[ $? -eq 0 && "$out" == *'"is_error":false'* ]]; then
      usage="$(sed -n 's/.*"input_tokens":\([0-9]*\).*"output_tokens":\([0-9]*\).*/in=\1 out=\2/p' <<<"$out" | head -1)"
      result="$(sed -n 's/.*"result":"\([^"]*\)".*/\1/p' <<<"$out" | head -1)"
      log "ping OK (attempt $attempt) ${usage:-} reply=\"${result:-}\" — session window open until $(date -d "+5 hours" '+%H:%M')"
      date '+%Y-%m-%d' >"$LAST_FILE"
      return 0
    fi
    log "ping FAILED (attempt $attempt): $(tr '\n' ' ' <<<"$out" | cut -c1-300)"
    ((attempt < 3)) && sleep $((attempt * 120))
  done
  return 1
}

# ----------------------------------------------------------- the daemon loop --
next_target() {  # epoch seconds of the next HH:MM that is still in the future
  local t
  t="$(date -d "today $PING_HOUR:$PING_MINUTE" +%s)"
  (( t <= $(date +%s) )) && t="$(date -d "tomorrow $PING_HOUR:$PING_MINUTE" +%s)"
  echo "$t"
}

run_loop() {
  log "daemon started (pid $$) — daily ping at $PING_HOUR:$PING_MINUTE, model $MODEL"

  # Catch-up: machine may have been off/asleep at the scheduled time.
  local now today
  now="$(date +%s)"; today="$(date '+%Y-%m-%d')"
  if (( CATCHUP_MINUTES > 0 )) && [[ "$(cat "$LAST_FILE" 2>/dev/null)" != "$today" ]]; then
    local scheduled; scheduled="$(date -d "today $PING_HOUR:$PING_MINUTE" +%s)"
    if (( now > scheduled && now - scheduled <= CATCHUP_MINUTES * 60 )); then
      log "missed today's slot by $(( (now - scheduled) / 60 )) min — sending catch-up ping"
      send_ping
    fi
  fi

  while true; do
    local target; target="$(next_target)"
    log "next ping at $(date -d "@$target" '+%Y-%m-%d %H:%M')"
    # Sleep in short chunks so suspend/resume and clock changes can't overshoot.
    while (( $(date +%s) < target )); do
      local left=$(( target - $(date +%s) ))
      sleep $(( left > 300 ? 300 : left ))
    done
    if [[ "$(cat "$LAST_FILE" 2>/dev/null)" == "$(date '+%Y-%m-%d')" ]]; then
      log "already pinged today, skipping"
    else
      send_ping
    fi
    sleep 60   # step past the target minute before recomputing
  done
}

# ------------------------------------------------------------- subcommands ---
is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null && return 0
  systemctl --user is-active --quiet claude-daily-ping.service 2>/dev/null
}

cmd_start() {
  if is_running; then
    echo "already running (pid $(cat "$PID_FILE" 2>/dev/null || echo "systemd"))"; return 0
  fi
  [[ -x "$CLAUDE_BIN" ]] || { echo "claude CLI not found at $CLAUDE_BIN"; return 1; }
  setsid nohup "$SELF" __loop >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  sleep 1
  is_running && echo "started (pid $(cat "$PID_FILE" 2>/dev/null || echo "systemd")) — next ping $PING_HOUR:$PING_MINUTE, log: $LOG_FILE"
}

cmd_stop() {
  if is_running; then
    kill "$(cat "$PID_FILE")" && echo "stopped (pid $(cat "$PID_FILE" 2>/dev/null || echo "systemd"))"
    log "daemon stopped"
  else
    echo "not running"
  fi
  rm -f "$PID_FILE"
}

cmd_status() {
  if is_running; then
    echo "running (pid $(cat "$PID_FILE" 2>/dev/null || echo "systemd"))"
  else
    echo "not running"
  fi
  echo "schedule:  every day at $PING_HOUR:$PING_MINUTE  (session then resets ~5h later)"
  echo "last ping: $(cat "$LAST_FILE" 2>/dev/null || echo never)"
  echo "log:       $LOG_FILE"
  [[ -f "$LOG_FILE" ]] && tail -3 "$LOG_FILE"
}

# Autostart via a systemd *user* service + linger, so the daemon comes up at
# boot and keeps running even when nobody is logged into a desktop session.
UNIT_FILE="$HOME/.config/systemd/user/claude-daily-ping.service"
cmd_autostart() {
  mkdir -p "$(dirname "$UNIT_FILE")"
  cat >"$UNIT_FILE" <<EOF
[Unit]
Description=Claude Code daily ping (opens the usage session at $PING_HOUR:$PING_MINUTE)

[Service]
Type=simple
ExecStart=$SELF __loop
Restart=always
RestartSec=30

[Install]
WantedBy=default.target
EOF
  cmd_stop >/dev/null 2>&1
  systemctl --user daemon-reload
  systemctl --user enable --now claude-daily-ping.service
  loginctl enable-linger "$USER" 2>/dev/null \
    && echo "linger enabled — runs at boot, no login needed" \
    || echo "note: could not enable linger (run: sudo loginctl enable-linger $USER)"
  echo "autostart installed: $UNIT_FILE"
  systemctl --user --no-pager status claude-daily-ping.service | head -5
}
cmd_unautostart() {
  systemctl --user disable --now claude-daily-ping.service 2>/dev/null
  rm -f "$UNIT_FILE"
  systemctl --user daemon-reload
  echo "autostart removed"
}

case "${1:-start}" in
  start)        cmd_start ;;
  stop)         cmd_stop ;;
  restart)      cmd_stop; cmd_start ;;
  status)       cmd_status ;;
  now)          send_ping && tail -1 "$LOG_FILE" ;;
  log)          tail -f "$LOG_FILE" ;;
  autostart)    cmd_autostart ;;
  unautostart)  cmd_unautostart ;;
  __loop)       run_loop ;;   # internal: the backgrounded loop
  *)            sed -n '4,20p' "$SELF"; exit 1 ;;
esac
