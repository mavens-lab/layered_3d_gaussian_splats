conda create --prefix /scratch/jiasi_root/jiasi98/shared_env/lightgaussian -y python=3.8
git clone https://github.com/lkeab/gaussian-grouping.git --recursive
cd gaussian-grouping
srun --jobid=14882183 --pty bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
source activate /scratch/jiasi_root/jiasi98/shared_env/lightgaussian
module load cuda/11.6.2 cudnn/11.6-v8.4.1
pip install submodules/compress-diff-gaussian-rasterization
# pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

#tensorboard --logdir OUTPUT/PATH/train
#srun --jobid=15644704 --pty bash
#tmux new -s light
#tmux a -t light
#salloc --job-name=light --cpus-per-task=1 --nodes=1 --ntasks-per-node=2 --mem-per-cpu=10G --time=14-00:00:00 --account=jiasi0 --partition=spgpu --gres=gpu:2


# #
# lit-gpt                  /home/zxuechen/.conda/envs/lit-gpt
#                          /scratch/jiasi_root/jiasi98/shared_env/LoG
#                          /scratch/jiasi_root/jiasi98/shared_env/gaussian_grouping
#                          /scratch/jiasi_root/jiasi98/shared_env/lightgaussian
#                          /scratch/jiasi_root/jiasi98/shared_env/nerfstudio
#                          /scratch/jiasi_root/jiasi98/shared_env/zoology
#                          /scratch/jiasi_root/jiasi98/shared_env/zoology2
# base                  *  /sw/pkgs/arc/python3.9-anaconda/2021.11