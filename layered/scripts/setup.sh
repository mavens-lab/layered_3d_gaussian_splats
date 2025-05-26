conda create lightgaussian -y python=3.8
git clone https://github.com/lkeab/gaussian-grouping.git --recursive
cd gaussian-grouping
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
source activate /scratch/jiasi_root/jiasi98/shared_env/lightgaussian
module load cuda/11.6.2 cudnn/11.6-v8.4.1
pip install submodules/compress-diff-gaussian-rasterization
# pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

