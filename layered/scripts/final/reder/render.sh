python render_video.py --source_path /home/zxuechen/LightGaussian/data/tandt/train --model_path /home/zxuechen/LightGaussian/pretrain_3d/train --skip_train --skip_test --video

python render_video_savepath.py --source_path /home/zxuechen/LightGaussian/data/tandt/train --model_path /home/zxuechen/LightGaussian/pretrain_3d/train  --skip_train --skip_test --video

python render_loadpath.py --source_path /home/zxuechen/LightGaussian/data/tandt/train --model_path /home/zxuechen/LightGaussian/pretrain_3d/train --trace_path /home/zxuechen/LightGaussian/result/trace/head_positions.txt --trace_type real --skip_train --skip_test --trace

python post2splat.py --source_path /scratch/jiasi_root/jiasi98/zxuechen/tandt/tandt/train --model_path /scratch/jiasi_root/jiasi98/zxuechen/pretrain_3d/train --trace_path /home/zxuechen/LightGaussian/ellipse_array.txt --trace_type synthetic --skip_train --skip_test --trace


python render_loadpath_sorting.py --source_path /home/zxuechen/LightGaussian/data/nerf360/bicycle --model_path /home/zxuechen/LightGaussian/result/model/bicycle/bicycle_135000_180000_v_important_score_sort_progressive_from180k --trace_path /home/zxuechen/LightGaussian/result/trace/head_positions.txt --trace_type real --skip_train --skip_test --trace

python concat_image_video.py
python render_loadpath_concat.py


python render_loadpath_sorting_1m.py --source_path /home/zxuechen/LightGaussian/data/nerf360/bicycle --model_path /home/zxuechen/LightGaussian/pretrain_3d/bicycle --trace_path /home/zxuechen/LightGaussian/result/trace/head_positions.txt --trace_type real --skip_train --skip_test --trace

python render_loadpath_concat.py --source_path /home/zxuechen/LightGaussian/data/nerf360/bicycle --model_path /home/zxuechen/LightGaussian/result/model/bicycle/bicycle_135000_180000_v_important_score_sort_progressive_from180k --trace_path /home/zxuechen/LightGaussian/result/trace/head_positions.txt --trace_type real --skip_train --skip_test --trace