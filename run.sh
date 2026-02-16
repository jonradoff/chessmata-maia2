#!/bin/bash
# Wrapper script for Maia2 Agent
# Automatically restarts the agent if it crashes.
# To stop completely: kill this wrapper script (kill $$PID)
# Killing the inner Python process will cause an automatic restart.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Clone the Maia2 engine if not present
if [ ! -d "maia2-engine" ]; then
    echo "Cloning Maia2 engine..."
    git clone https://github.com/CSSLab/maia2.git maia2-engine
fi

echo "============================================"
echo "  Maia2 Agent Wrapper"
echo "  PID: $$"
echo "  To stop: kill $$"
echo "============================================"
echo ""

RESTART_DELAY=5

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Maia2 agent..."
    python3 -m agent.main "$@"
    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Agent exited with code $EXIT_CODE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting in ${RESTART_DELAY} seconds..."
    sleep "$RESTART_DELAY"
done
