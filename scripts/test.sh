
# NV_NAME="unirgb-ir"
# source activate $ENV_NAME || conda activate $ENV_NAME
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH



CUDA_HOME=/usr/local/cuda-11.6 CUDA_VISIBLE_DEVICES=0 \
tools/dist_test.sh \
configs/GDINO/FLIR/grounding_dino_swin-t_finetune_16xb2_1x_RGBT_flir_illum.py \
path/to/checkpoint 1 --work-dir test \
