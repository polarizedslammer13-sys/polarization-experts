#!/bin/bash
# PTA-A orchestrator. Designed to be run inside tmux.
#
# Layout when launching:
#   tmux new -s pta-a -d 'bash /root/polarization-experts/pta_a/run.sh'
#   tmux new-window -t pta-a 'watch -n 5 nvidia-smi'
#   tmux new-window -t pta-a 'tail -F /root/polarization-experts/pta_a/logs/_orchestrator.log'
#   tmux attach -t pta-a
#
# Resume semantics:
#   - stokes.py checks results.json and skips completed (mode, seed)
#   - This script can be re-launched after interrupt; it just re-iterates
#
# Pre-flight:
#   - Verifies norm_stats.json exists
#   - Verifies GPU visible via nvidia-smi
#   - Fails fast on any per-run non-zero exit (set -e)

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"

MODES=("A0" "A1" "A2" "A2-raw" "A3")
SEEDS=(1 2 3 4 5)

LOG_DIR="logs"
ORCH_LOG="${LOG_DIR}/_orchestrator.log"
mkdir -p "$LOG_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }

log() { echo "[$(ts)] $*" | tee -a "$ORCH_LOG"; }

# ----- pre-flight -----
log "==== PTA-A orchestrator start ===="
log "cwd: $(pwd)"

if [ ! -f norm_stats.json ]; then
    log "FATAL: norm_stats.json missing. Run: python3 norm_stats.py"
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "FATAL: nvidia-smi not found"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1 || true)
if [ -z "$GPU_NAME" ]; then
    log "FATAL: no GPU detected"
    exit 1
fi
log "GPU detected: $GPU_NAME"

TOTAL=$(( ${#MODES[@]} * ${#SEEDS[@]} ))
log "Planned: ${#MODES[@]} modes × ${#SEEDS[@]} seeds = $TOTAL runs"

# ----- run loop -----
DONE_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        tag="${mode}_seed${seed}"
        run_log="${LOG_DIR}/${tag}.log"

        # Ask the trainer if this (mode, seed) is already complete.
        # We let the trainer itself decide (single source of truth).
        log "---- START $tag ----"
        if python3 stokes.py --mode "$mode" --seed "$seed" > "$run_log" 2>&1; then
            if grep -q "^\[SKIP\]" "$run_log"; then
                log "SKIP  $tag  (already in results)"
                SKIP_COUNT=$((SKIP_COUNT+1))
            else
                # Extract a one-line summary
                summary=$(grep -E "^Test  PCC|^Best Val PCC" "$run_log" | tr '\n' ' | ' || true)
                log "DONE  $tag  ${summary:-(no summary)}"
                DONE_COUNT=$((DONE_COUNT+1))
            fi
        else
            log "FAIL  $tag  (see $run_log)"
            FAIL_COUNT=$((FAIL_COUNT+1))
            # Don't continue on failure — fail fast so we can investigate
            exit 1
        fi
    done
done

log "==== Orchestrator finished ===="
log "  completed this session : $DONE_COUNT"
log "  skipped (already done) : $SKIP_COUNT"
log "  failed                 : $FAIL_COUNT"
log "  results file           : $(pwd)/results.json"
log "Next: python3 analyze.py"
