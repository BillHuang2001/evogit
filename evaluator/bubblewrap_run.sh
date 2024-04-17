#!/usr/bin/env bash
# bubblewrap_run.sh
IFS=',' read -r -a GPUS <<< $1
shift
for index in "${!GPUS[@]}"
do
    GPU=${GPUS[index]}
    DEV_BIND_ARGS+="--dev-bind /dev/nvidia$GPU /dev/nvidia$GPU "
done
if [[ -n $DEV_BIND_ARGS ]]; then
    DEV_BIND_ARGS="--dev /dev --dev-bind /dev/nvidiactl /dev/nvidiactl --dev-bind /dev/nvidia-uvm /dev/nvidia-uvm $DEV_BIND_ARGS"
fi
bwrap \
    --ro-bind / / \
    $DEV_BIND_ARGS \
    $@
