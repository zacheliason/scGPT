#!/bin/bash

#SBATCH -o /CHILDRENS/home/g29755807/scgpt/runs/%j/scgpt.log
#SBATCH -e /CHILDRENS/home/g29755807/scgpt/runs/%j/scgpt.err
#SBATCH -p med-gpu
#SBATCH -N 1
#SBATCH -D /scratch/ligrp/zach/scgpt
#SBATCH -J scGPT_Test
#SBATCH --export=None
#SBATCH -t 4:00:00
#SBATCH --mem=30G

HOME_DIR=/CHILDRENS/home/g29755807/scgpt
RUN_DIR=$HOME_DIR/runs/${SLURM_JOB_ID}/
mkdir -p $RUN_DIR

WD=/scratch/ligrp/zach/scgpt
DATA_DIR=$WD/data

if [ -d "$WD" ]; then
    echo "Directory $WD exists. Aborting setup."
else
    # Set up working directory
    echo "Directory $WD does not exist. Initiating setup."
    mkdir -p $WD

    echo "Created working directory $WD"
    cd $WD
    #cp $HOME_DIR/pyproject.toml $WD/pyproject.toml
    #cp $HOME_DIR/scgpt_perturb.py $WD/scgpt_perturb.py

    # Download scGPT via uv pip
    git clone https://github.com/zacheliason/scGPT.git $WD/scGPT
    mv $WD/scGPT/* $WD/
    rm -rf $WD/scGPT

    # Set up uv
    export UV_ROOT=$WD/.uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=$UV_ROOT/bin:$PATH

    # Create virtual environment
    uv python install 3.10
    uv venv $WD/.venv
    uv pip install -r pyproject.toml
    source $WD/.venv/bin/activate

    # Download data
    mkdir -p $DATA_DIR
    gdown --folder https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y -O $DATA_DIR/scGPT_human

    echo "Download complete. Files saved to: $DATA_DIR"
fi

echo "Running scGPT on data in $DATA_DIR and saving results to $RUN_DIR"
source $WD/.venv/bin/activate && python3 scgpt_perturb_map.py -d $DATA_DIR -o $RUN_DIR