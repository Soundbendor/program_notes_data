#!/bin/bash
#SBATCH -J extraction_5000
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH -t 1-23:59:59
#SBATCH --gres=gpu:2
#SBATCH --export=ALL
python RAG_rerank.py