#!/bin/bash

# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=10000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
   $2 < threshold { print $1; exit }
  '
}

# Initial port number
port=7245
# This is an example script to load from ply file.
# Only one dataset specified here, but you could run multiple
declare -a run_args=(
    "bicycle"
    "bonsai"
    "counter"
    # "kitchen"
    # "room"
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
declare -a target_num_save=(45000)
declare -a target_num_train=(90000)
# decay rate for the following prune
declare -a prune_decays=(1)
# The volumetric importance power. The higher it is the more weight the volume is in the Global significant
declare -a v_pow=(0.1)

# prune type, by default the Global significant listed in the paper, but there are other option that you can play with
declare -a prune_types=(
  "v_important_score"
  # "important_score"
  # "count"
  # "opacity"
  )

# echo ${#target_num[@]}
# echo ${#prune_decays[@]}
# echo ${#v_pow[@]}
# echo "${#target_num[@]}" -ne "${#prune_decays[@]}" 
# echo "${#prune_percents[@]}" -ne "${#v_pow[@]}"
# Check that prune_percents, prune_decays, and v_pow arrays have the same length
# if [ "${#target_num[@]}" -ne "${#prune_decays[@]}" ] || [ "${#target_num[@]}" -ne "${#v_pow[@]}" ]; then
#   echo "The lengths of target_num, prune_decays, and v_pow arrays do not match."
#   exit 1
# fi
# /ssd1/zhiwen/projects/compress_gaussian/output2/bicycle/point_cloud/iteration_30000/point_cloud.ply
# Loop over the arguments array
for arg in "${run_args[@]}"; do
  prune_decay="${prune_decays[0]}"
  prune_type="${prune_types[0]}"
  vp="${v_pow[0]}"
  python prune_finetune_sorting_progressive.py \
    -s "/scratch/jiasi_root/jiasi98/zxuechen/nerf360/$arg" \
    -m "/scratch/jiasi_root/jiasi98/zxuechen/OUTPUT/PATH/${arg}/${arg}_${target_num_save}_${target_num_train}_${prune_type}_sort_progressive_from180k" \
    --eval \
    --port $port \
    --start_pointcloud_sort "/scratch/jiasi_root/jiasi98/zxuechen/OUTPUT/PATH/baseline/sort/${arg}_v_important_score_ply_sort_all_from180k/point_cloud/iteration_3/point_cloud.ply" \
    --load_pointcloud_sort "/scratch/jiasi_root/jiasi98/zxuechen/OUTPUT/PATH/seperate/${arg}_45000/point_cloud/iteration_5000/point_cloud.ply" \
    --iteration 5000 \
    --prune_iterations 2 \
    --target_num_save 45000 \
    --target_num_train 90000 \
    --prune_type $prune_type \
    --prune_decay $prune_decay \
    --position_lr_init 0.00005 \
    --position_lr_max_steps 5000 \
    --v_pow $vp > "logs_train/${arg}_${target_num_save}_${target_num_train}_${prune_type}_ply_sort_progressive .log" 2>&1
  ((port++))
  python prune_finetune_sorting_progressive.py \
    -s "/scratch/jiasi_root/jiasi98/zxuechen/nerf360/$arg" \
    -m "/scratch/jiasi_root/jiasi98/zxuechen/OUTPUT/PATH/${arg}/${arg}_90000_135000_${prune_type}_sort_progressive_from180k" \
    --eval \
    --port $port \
    --start_pointcloud_sort "/scratch/jiasi_root/jiasi98/zxuechen/OUTPUT/PATH/baseline/sort/${arg}_v_important_score_ply_sort_all_from180k/point_cloud/iteration_3/point_cloud.ply" \
    --load_pointcloud_sort "/scratch/jiasi_root/jiasi98/zxuechen/OUTPUT/PATH/${arg}/${arg}_45000_90000_v_important_score_sort_progressive_from180k/point_cloud/iteration_5000/point_cloud.ply" \
    --iteration 5000 \
    --prune_iterations 2 \
    --target_num_save 90000 \
    --target_num_train 135000 \
    --prune_type $prune_type \
    --prune_decay $prune_decay \
    --position_lr_init 0.00005 \
    --position_lr_max_steps 5000 \
    --v_pow $vp > "logs_train/${arg}_90000_135000_${prune_type}_ply_sort_progressive .log" 2>&1
  ((port++))
  python prune_finetune_sorting_progressive.py \
    -s "/scratch/jiasi_root/jiasi98/zxuechen/nerf360/$arg" \
    -m "/scratch/jiasi_root/jiasi98/zxuechen/OUTPUT/PATH/${arg}/${arg}_135000_180000_${prune_type}_sort_progressive_from180k" \
    --eval \
    --port $port \
    --start_pointcloud_sort "/scratch/jiasi_root/jiasi98/zxuechen/OUTPUT/PATH/baseline/sort/${arg}_v_important_score_ply_sort_all_from180k/point_cloud/iteration_3/point_cloud.ply" \
    --load_pointcloud_sort "/scratch/jiasi_root/jiasi98/zxuechen/OUTPUT/PATH/${arg}/${arg}_90000_135000_v_important_score_sort_progressive_from180k/point_cloud/iteration_5000/point_cloud.ply" \
    --iteration 5000 \
    --prune_iterations 2 \
    --target_num_save 135000 \
    --target_num_train 180000 \
    --prune_type $prune_type \
    --prune_decay $prune_decay \
    --position_lr_init 0.00005 \
    --position_lr_max_steps 5000 \
    --v_pow $vp > "logs_train/${arg}_135000_180000_${prune_type}_ply_sort_progressive .log" 2>&1


  ((port++))

done
wait


echo "All prune_finetune.py runs completed."

#/scratch/jiasi_root/jiasi98/zxuechen/pretrain_3d/$arg/point_cloud/iteration_30000/point_cloud.ply