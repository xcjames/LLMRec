#!/bin/bash
#PBS -N XuChenJobsGetPub
#PBS -l select=1:ngpus=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -P personal-chen035
#PBS -q normal
#PBS -o ./output_log/


#select=1:ncpus=1:mem=1G 
#select=1:ngpus=1
#select=8:ncpus=128:mem=440G:mpiprocs=128:ompthreads=1
#select=1:ncpus=16:mem=55G:mpiprocs=16:ompthreads=1
#-o ./output_log/out_\\$PBS_JOBID_\${(date+%Y%m%d-%H%M%S)}.out

########################### Real-time Output file. ###############################
exec 1>/home/users/ntu/chen035/scratch/LC-Rec/output_log/${PBS_JOBID}_o.stdout
exec 2>/home/users/ntu/chen035/scratch/LC-Rec/output_log/${PBS_JOBID}_e.stderr

########################### Load env ###############################
# Load necessary modules (if your cluster uses modules)
cd ${PBS_O_WORKDIR}
module load miniforge3 # Only if Anaconda is installed via modules
module load gcc/11.2.0 
module swap PrgEnv-cray/8.3.3 PrgEnv-gnu/8.3.3

# conda env list
# conda list -n LC-Rec
conda activate LC-Rec

# conda install libgcc
# conda install pyarrow --update
# # Create the Conda environment (only needed once)
# conda create -n LC-Rec python=3.8 -y

# # Install PyTorch 1.13.1 with CUDA 11.7


# pip install torch==1.13.1+cu117 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
# pip install transformers==4.37.1
# # conda install conda-forge::transformers
# # # Install additional packages

# # conda install conda-forge::transformers
# pip install accelerate bitsandbytes deepspeed evaluate peft sentencepiece tqdm

# # # Verify installations
# python -c "
# import torch
# import transformers
# import accelerate
# import deepspeed
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
# print(transformers.__version__)
# print(accelerate.__version__)
# print(deepspeed.__version__)
# "
# python --version
# ######################## Model Checkpoint (load LC-Rec delta to base model) #################################
# export HF_ENDPOINT="https://hf-mirror.com/"

# # python /home/users/ntu/chen035/scratch/LC-Rec/convert/merge_delta.py \
# #     --base-model-path huggyllama/llama-7b \
# #     --target-model-path /home/users/ntu/chen035/scratch/LC-Rec/target_model_path \
# #     --delta-path bwzheng0324/lc-rec-games-delta

# pwd
# gcc --version

######################## Model Training #######################################
export HF_ENDPOINT="https://hf-mirror.com/"
DATASET=Games
BASE_MODEL=huggyllama/llama-7b
DATA_PATH=./data
OUTPUT_DIR=./ckpt/$DATASET/

torchrun --nproc_per_node=1 --master_port=23324 finetune.py \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --epochs 2 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --deepspeed ./config/ds_z3_bf16.json \
    --bf16 \
    --only_train_response \
    --tasks seqrec,item2index,index2item,fusionseqrec,itemsearch,preferenceobtain \
    --train_prompt_sample_num 1,1,1,1,1,1 \
    --train_data_sample_num 0,0,0,100000,0,0 \
    --index_file .index.json


cd convert
nohup ./convert.sh $OUTPUT_DIR >convert.log 2>&1 &
cd ..


######################## Deactivate Conda environment
conda deactivate