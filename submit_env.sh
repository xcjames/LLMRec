#!/bin/bash
#PBS -N XuChenJobsGetPub
#PBS -l select=1:ncpus=16:mem=55G:mpiprocs=16:ompthreads=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -P personal-chen035
#PBS -q normal
#PBS -o ./output_log/


#select=1:ncpus=1:mem=1G 
#select=1:ngpus=1
#select=8:ncpus=128:mem=440G:mpiprocs=128:ompthreads=1
#-o ./output_log/out_\\$PBS_JOBID_\${(date+%Y%m%d-%H%M%S)}.out

exec 1>file.stdout
exec 2>file.stderr

# Load necessary modules (if your cluster uses modules)
module load miniforge3  # Only if Anaconda is installed via modules
conda env list

# conda list -n LC-Rec
conda activate LC-Rec
# conda install conda-forge::transformers


# # Create the Conda environment (only needed once)
# conda create -n LC-Rec python=3.8 -y

# # Install PyTorch 1.13.1 with CUDA 11.7
pip install torch==1.13.1+cu117 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# # Install additional packages

# conda install conda-forge::transformers
# pip install accelerate bitsandbytes deepspeed evaluate peft sentencepiece tqdm

# # Verify installations
python -c "
import torch
import transformers
import accelerate
import deepspeed
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(transformers.__version__)
print(accelerate.__version__)
print(deepspeed.__version__)
"

# # Deactivate Conda environment
conda deactivate