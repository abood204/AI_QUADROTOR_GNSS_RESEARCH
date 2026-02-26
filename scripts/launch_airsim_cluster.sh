#!/usr/bin/env bash
# launch_airsim_cluster.sh — Launch N+1 AirSim instances for parallel training.
#
# Each instance gets a unique HOME directory with a patched settings.json
# so they bind to different API ports (base_port + i).
#
# Usage:
#   ./scripts/launch_airsim_cluster.sh 4                # 4 train + 1 eval
#   ./scripts/launch_airsim_cluster.sh 4 --stop         # kill all instances
#   ./scripts/launch_airsim_cluster.sh 4 --status       # check which are running
#
# Requires: AIRSIM_BIN environment variable pointing to the AirSim binary.
#   export AIRSIM_BIN=/path/to/AirSimNH/LinuxNoEditor/AirSimNH.sh

set -euo pipefail

NUM_TRAIN=${1:?Usage: launch_airsim_cluster.sh NUM_TRAIN_ENVS [--stop|--status]}
ACTION=${2:-start}
BASE_PORT=${BASE_PORT:-41451}
SETTINGS_TEMPLATE=${SETTINGS_TEMPLATE:-configs/settings_training.json}
AIRSIM_BIN=${AIRSIM_BIN:?Set AIRSIM_BIN to your AirSim binary path}

# Total instances = NUM_TRAIN + 1 eval
TOTAL=$((NUM_TRAIN + 1))

kill_all() {
    echo "[cluster] Stopping all AirSim instances..."
    for i in $(seq 0 $((TOTAL - 1))); do
        pidfile="/tmp/airsim_home_${i}/airsim.pid"
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" && echo "  Killed instance $i (PID $pid)"
            fi
            rm -f "$pidfile"
        fi
    done
    echo "[cluster] All instances stopped."
}

status_all() {
    echo "[cluster] AirSim instance status:"
    for i in $(seq 0 $((TOTAL - 1))); do
        port=$((BASE_PORT + i))
        pidfile="/tmp/airsim_home_${i}/airsim.pid"
        if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
            echo "  Instance $i: RUNNING (port $port, PID $(cat "$pidfile"))"
        else
            echo "  Instance $i: STOPPED (port $port)"
        fi
    done
}

launch_all() {
    echo "[cluster] Launching $TOTAL AirSim instances ($NUM_TRAIN train + 1 eval)..."
    echo "[cluster] Base port: $BASE_PORT, template: $SETTINGS_TEMPLATE"

    if [ ! -f "$SETTINGS_TEMPLATE" ]; then
        echo "ERROR: Settings template not found: $SETTINGS_TEMPLATE" >&2
        exit 1
    fi

    for i in $(seq 0 $((TOTAL - 1))); do
        port=$((BASE_PORT + i))
        home_dir="/tmp/airsim_home_${i}"
        settings_dir="${home_dir}/Documents/AirSim"

        mkdir -p "$settings_dir"

        # Patch ApiServerPort in the settings copy
        sed "s/\"ApiServerPort\": [0-9]*/\"ApiServerPort\": ${port}/" \
            "$SETTINGS_TEMPLATE" > "${settings_dir}/settings.json"

        # Launch with custom HOME so AirSim reads our patched settings
        if [ "$i" -eq "$((TOTAL - 1))" ]; then
            role="eval"
        else
            role="train-$i"
        fi

        echo "  Starting instance $i ($role) on port $port..."
        HOME="$home_dir" "$AIRSIM_BIN" -windowed -ResX=160 -ResY=120 &
        echo $! > "${home_dir}/airsim.pid"

        # Small delay between launches to avoid resource contention
        sleep 2
    done

    echo "[cluster] Waiting 10s for instances to initialize..."
    sleep 10

    # Verify ports are listening
    echo "[cluster] Verifying API ports..."
    all_ok=true
    for i in $(seq 0 $((TOTAL - 1))); do
        port=$((BASE_PORT + i))
        if command -v nc &>/dev/null; then
            if nc -z localhost "$port" 2>/dev/null; then
                echo "  Port $port: OK"
            else
                echo "  Port $port: NOT RESPONDING" >&2
                all_ok=false
            fi
        else
            echo "  Port $port: (nc not available, skipping check)"
        fi
    done

    if [ "$all_ok" = true ]; then
        echo "[cluster] All instances ready!"
    else
        echo "[cluster] WARNING: Some instances may not be ready yet. Retry in a few seconds."
    fi

    echo ""
    echo "Train command:"
    echo "  python -m src.training.train --num_envs $NUM_TRAIN --base_port $BASE_PORT --total_timesteps 4096"
}

case "$ACTION" in
    --stop)  kill_all   ;;
    --status) status_all ;;
    *)       launch_all  ;;
esac
