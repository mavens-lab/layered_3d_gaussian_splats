#!/bin/bash

# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=10000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
   $2 < threshold { print $1; exit }
  '
}

# Initial port number
port=6045
# This is an example script to load from ply file.
# Only one dataset specified here, but you could run multiple
declare -a run_args=(
    "bicycle"
    "bonsai"
    "counter"
    "kitchen"
    "room"
    # "stump"
    # "garden"
    # "train"
    # "truck"
    # "chair"
    # "drums"
    # "ficus"
    # "hotdog"
    # "lego"
    # "mic"
    # "materials"
    # "ship"
  )


# Prune percentages and corresponding decays, volume power
declare -a target_num=(
  6000
  45000
  90000
  135000
  180000)
# decay rate for the following prune
declare -a prune_decays=(1)
# The volumetric importance power. The higher it is the more weight the volume is in the Global significant
declare -a v_pow=(0.1)

# prune type, by default the Global significant listed in the paper, but there are other option that you can play with
declare -a prune_types=(
  "v_important_score"
  # "important_score"
  "count"
  "opacity"
  )


# # Check that prune_percents, prune_decays, and v_pow arrays have the same length
# if [ "${#target_num[@]}" -ne "${#prune_decays[@]}" ] || [ "${#target_num[@]}" -ne "${#v_pow[@]}" ]; then
#   echo "The lengths of prune_percents, prune_decays, and v_pow arrays do not match."
#   exit 1
# fi
# /ssd1/zhiwen/projects/compress_gaussian/output2/bicycle/point_cloud/iteration_30000/point_cloud.ply
# Loop over the arguments array
for arg in "${run_args[@]}"; do
  for i in "${!target_num[@]}"; do
    target_num="${target_num[i]}"
    prune_decay="${prune_decays[0]}"
    vp="${v_pow[0]}"

    for prune_type in "${prune_types[@]}"; do
      # Wait for an available GPU
      while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
          echo "GPU $gpu_id is available. Starting prune_finetune_save_sort.py with dataset '$arg', target_num '$target_num', prune_type '$prune_type', prune_decay '$prune_decay', and v_pow '$vp' on port $port"
        
          CUDA_VISIBLE_DEVICES=$gpu_id python prune_finetune_save_sort.py \
            -s "/scratch/jiasi_root/jiasi98/zxuechen/nerf360/$arg" \
            -m "/scratch/jiasi_root/jiasi98/zxuechen/OUTPUT/PATH/baseline/${arg}_${target_num}_${prune_type}_sort" \
            --eval \
            --port $port \
            --start_pointcloud "/scratch/jiasi_root/jiasi98/zxuechen/pretrain_3d/$arg/point_cloud/iteration_30000/point_cloud.ply" \
            --iteration 2 \
            --test_iterations 2 \
            --save_iterations 2 \
            --prune_iterations 1 \
            --target_num $target_num \
            --prune_type $prune_type \
            --prune_decay $prune_decay \
            --position_lr_init 0.000005 \
            --position_lr_max_steps 5000 \
            --v_pow $vp > "logs_prune/${arg}${target_num}_${prune_type}_ply_sort .log" 2>&1

          # Increment the port number for the next run
          ((port++))
          # Allow some time for the process to initialize and potentially use GPU memory
          sleep 60
          break
        else
          echo "No GPU available at the moment. Retrying in 1 minute."
          sleep 60
        fi
      done
    done
  done
done
wait
echo "All prune_finetune.py runs completed."
