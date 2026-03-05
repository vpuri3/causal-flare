#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/submit.sh [-p advanced|preempt|general] [-t 24h] [-n 1..8] [-N 1..8]

Cluster note:
  These Slurm defaults are configured for H100 GPUs on the Google Cloud A3 Mega cluster.

Options:
  -p  Partition to use. Default: preempt
  -t  Wall time. Default: 24h
  -n  GPUs per node. Default: 1
  -N  Number of nodes. Default: 1
  -h  Show this help message
EOF
}

partition="preempt"
time_limit="24h"
gpus_per_node=1
num_nodes=1

normalize_time_for_slurm() {
  local raw_time="$1"
  if [[ "${raw_time}" =~ ^([0-9]+)h$ ]]; then
    local hours="${BASH_REMATCH[1]}"
    printf '%d:00:00\n' "${hours}"
    return 0
  fi

  # Pass through values already in a Slurm-compatible form such as
  # MM, MM:SS, HH:MM:SS, or D-HH[:MM[:SS]].
  printf '%s\n' "${raw_time}"
}

time_to_seconds() {
  local raw_time="$1"
  local days=0
  local hours=0
  local minutes=0
  local seconds=0

  if [[ "${raw_time}" =~ ^([0-9]+)h$ ]]; then
    hours="${BASH_REMATCH[1]}"
    printf '%d\n' "$(( hours * 3600 ))"
    return 0
  fi

  # Slurm shorthand: MM
  if [[ "${raw_time}" =~ ^([0-9]+)$ ]]; then
    minutes="${BASH_REMATCH[1]}"
    printf '%d\n' "$(( minutes * 60 ))"
    return 0
  fi

  # Slurm shorthand: MM:SS
  if [[ "${raw_time}" =~ ^([0-9]+):([0-9]+)$ ]]; then
    minutes="${BASH_REMATCH[1]}"
    seconds="${BASH_REMATCH[2]}"
    printf '%d\n' "$(( minutes * 60 + seconds ))"
    return 0
  fi

  # Slurm shorthand: HH:MM:SS
  if [[ "${raw_time}" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
    hours="${BASH_REMATCH[1]}"
    minutes="${BASH_REMATCH[2]}"
    seconds="${BASH_REMATCH[3]}"
    printf '%d\n' "$(( hours * 3600 + minutes * 60 + seconds ))"
    return 0
  fi

  # Slurm shorthand: D-HH[:MM[:SS]]
  if [[ "${raw_time}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    days="${BASH_REMATCH[1]}"
    hours="${BASH_REMATCH[2]}"
    printf '%d\n' "$(( days * 86400 + hours * 3600 ))"
    return 0
  fi

  if [[ "${raw_time}" =~ ^([0-9]+)-([0-9]+):([0-9]+)$ ]]; then
    days="${BASH_REMATCH[1]}"
    hours="${BASH_REMATCH[2]}"
    minutes="${BASH_REMATCH[3]}"
    printf '%d\n' "$(( days * 86400 + hours * 3600 + minutes * 60 ))"
    return 0
  fi

  if [[ "${raw_time}" =~ ^([0-9]+)-([0-9]+):([0-9]+):([0-9]+)$ ]]; then
    days="${BASH_REMATCH[1]}"
    hours="${BASH_REMATCH[2]}"
    minutes="${BASH_REMATCH[3]}"
    seconds="${BASH_REMATCH[4]}"
    printf '%d\n' "$(( days * 86400 + hours * 3600 + minutes * 60 + seconds ))"
    return 0
  fi

  return 1
}

while getopts ":p:t:n:N:h" opt; do
  case "${opt}" in
    p)
      partition="${OPTARG}"
      ;;
    t)
      time_limit="${OPTARG}"
      ;;
    n)
      gpus_per_node="${OPTARG}"
      ;;
    N)
      num_nodes="${OPTARG}"
      ;;
    h)
      usage
      exit 0
      ;;
    :)
      echo "Error: option -${OPTARG} requires an argument." >&2
      usage
      exit 1
      ;;
    \?)
      echo "Error: invalid option -${OPTARG}" >&2
      usage
      exit 1
      ;;
  esac
done

case "${partition}" in
  advanced)
    qos="adv_4gpu_qos"
    ;;
  preempt)
    qos="preempt_qos"
    ;;
  general)
    qos="general_qos"
    ;;
  *)
    echo "Error: invalid partition '${partition}'. Use one of: advanced, preempt, general." >&2
    exit 1
    ;;
esac

if ! [[ "${gpus_per_node}" =~ ^[1-8]$ ]]; then
  echo "Error: invalid -n '${gpus_per_node}'. Allowed values: 1..8." >&2
  exit 1
fi

if ! [[ "${num_nodes}" =~ ^[1-8]$ ]]; then
  echo "Error: invalid -N '${num_nodes}'. Allowed values: 1..8." >&2
  exit 1
fi

if (( num_nodes > 1 )); then
  gpus_per_node=8
fi

slurm_time="$(normalize_time_for_slurm "${time_limit}")"

if [[ "${partition}" == "general" ]]; then
  if requested_seconds="$(time_to_seconds "${time_limit}")"; then
    if (( requested_seconds > 12 * 3600 )); then
      echo "Requested time '${time_limit}' exceeds general partition cap; clamping to 12h." >&2
      time_limit="12h"
      slurm_time="12:00:00"
    fi
  fi
fi

echo "Submitting job:"
echo "  partition: ${partition}"
echo "  qos: ${qos}"
echo "  time: ${time_limit} (sbatch: ${slurm_time})"
echo "  nodes: ${num_nodes}"
echo "  gpus/node: ${gpus_per_node}"
echo "  ntasks/node: ${gpus_per_node}"
echo "  cpus/task: 26"

sbatch \
  --account=lkara \
  --partition="${partition}" \
  --qos="${qos}" \
  --job-name=FLARE \
  --time="${slurm_time}" \
  --nodes="${num_nodes}" \
  --gres="gpu:${gpus_per_node}" \
  --ntasks-per-node="${gpus_per_node}" \
  --cpus-per-task=26 \
  --wrap='sleep 24h'

echo
echo "Current queue for user $(whoami):"
squeue -u "$(whoami)"
