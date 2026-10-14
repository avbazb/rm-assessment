#!/usr/bin/env bash
set -euo pipefail

taskroot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
action="${1:-launch}"
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in launch|build|verify|verify-display|check) ;; *) echo 'Use launch, build, verify, verify-display, or check' >&2; exit 2 ;; esac

set +u
source /opt/ros/humble/setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-41}" ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export GAZEBO_MASTER_URI="${GAZEBO_MASTER_URI:-http://127.0.0.1:11381}"
export GAZEBO_MODEL_DATABASE_URI=''
export QT_X11_NO_MITSHM=1
cache="${XDG_CACHE_HOME:-$HOME/.cache}/rm-assessment"
mkdir -p "$cache"
if [[ -z "${RMUC_MODEL_DIR:-}" && -f "$cache/map-path.txt" ]]; then
  IFS= read -r RMUC_MODEL_DIR < "$cache/map-path.txt"
fi
export RMUC_MODEL_DIR="${RMUC_MODEL_DIR:-$cache/models/rmuc2026_v20_native}"
python3 "$taskroot/check_map.py" "$RMUC_MODEL_DIR"
RMUC_MODEL_DIR="$(realpath -- "$RMUC_MODEL_DIR")"
export RMUC_MODEL_DIR
printf '%s\n' "$RMUC_MODEL_DIR" > "$cache/map-path.txt"
export GAZEBO_MODEL_PATH="$(dirname -- "$RMUC_MODEL_DIR"):${GAZEBO_MODEL_PATH:-}"
if [[ "$action" == check ]]; then exit 0; fi

ws="${RMUC_WORKSPACE:-$cache/ros2_ws}"
package="$taskroot/src/rmuc26_assessment"
mkdir -p "$ws/src"
if [[ -L "$ws/src/rmuc26_assessment" ]]; then
  if [[ "$(readlink -f -- "$ws/src/rmuc26_assessment")" != "$package" ]]; then
    echo 'Workspace belongs to another checkout. Set RMUC_WORKSPACE to a new directory.' >&2
    exit 2
  fi
elif [[ -e "$ws/src/rmuc26_assessment" ]]; then
  echo 'Workspace contains another package. Set RMUC_WORKSPACE to a new directory.' >&2
  exit 2
else
  ln -s -- "$package" "$ws/src/rmuc26_assessment"
fi
if [[ "$action" == build || ! -f "$ws/install/rmuc26_assessment/share/rmuc26_assessment/worlds/rmuc26.world" ]]; then
  (cd -- "$ws" && colcon build --symlink-install --packages-select rmuc26_assessment)
fi
set +u
source "$ws/install/setup.bash"
set -u
case "$action" in
  build) ;;
  launch) exec ros2 launch rmuc26_assessment assessment.launch.py "$@" ;;
  verify) exec python3 "$taskroot/verify.py" "$@" ;;
  verify-display) exec python3 "$taskroot/verify_display.py" "$@" ;;
esac
