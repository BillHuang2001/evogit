#!/usr/bin/env bash
# bubblewrap_run.sh

usage() {
    echo "Usage: bubblewrap_run.sh [--nv=<nvidia-gpu-ids>] args..."
}

args=()
proc_args=true
for arg in "$@"; do
    if $proc_args; then
        case $arg in
            --nv=*)
                gpu_ids="${arg#*=}"
                IFS=',' read -r -a nvidia_gpus <<< $gpu_ids
            ;;
            --help)
                usage
                exit 0
            ;;
            *)
                args["${#args[*]}"]=$arg
                proc_args=false
            ;;
        esac
    else
        args["${#args[*]}"]=$arg
    fi
done

for index in "${!nvidia_gpus[@]}"
do
    GPU=${nvidia_gpus[index]}
    DEV_BIND_ARGS+="--dev-bind /dev/nvidia$GPU /dev/nvidia$GPU "
done
if [[ -n $DEV_BIND_ARGS ]]; then
    DEV_BIND_ARGS="--dev /dev --dev-bind /dev/nvidiactl /dev/nvidiactl --dev-bind /dev/nvidia-uvm /dev/nvidia-uvm $DEV_BIND_ARGS "
fi
bwrap \
    --ro-bind /usr /usr \
    --ro-bind /home /home \
    --symlink usr/lib /lib \
    --symlink usr/lib64 /lib64 \
    --symlink usr/bin /bin \
    --symlink usr/sbin /sbin \
    --tmpfs /tmp \
    --die-with-parent \
    --unshare-all \
    --hostname sandbox \
    $DEV_BIND_ARGS \
    ${args[@]}
