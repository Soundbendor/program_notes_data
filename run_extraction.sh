#!/bin/bash
#SBATCH -J 13000_extraction_14000
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH -t 31:59:59
#SBATCH --gres=gpu:2
#SBATCH --export=ALL
python program_note_extraction_llama3.1.py
