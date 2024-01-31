#!/bin/bash

#SBATCH --job-name=caption_upsample
#SBATCH --output=caption_upsample.out.%j
#SBATCH --error=caption_upsample.out.%j
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=6:00:00
#SBATCH --account=stability
#SBATCH --partition=a40

python upsample_captions.py