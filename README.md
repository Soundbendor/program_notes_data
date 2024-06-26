Install miniconda if you don't have it:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

chmod a+x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh

Note - While going through Conda’s guided installer, it will ask you to specify an installation directory. This will default to your home directory - which has a 14gb disk restriction as a student. You will likely want to  change this path to use the SoundBendOR Lab filespace. /nfs/guille/eecs_research/soundbendor/<ONID>

# Create a new conda environment and activate it
conda create -n program_notes python=3.12

conda activate program_notes


pip install jupyterlab

module load gcc cmake cuda/11.8 slurm

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

If you get this error and can’t open Jupyter Session:
AttributeError: partially initialized module 'charset_normalizer' has no attribute 'md__mypyc' (most likely due to a circular import)
Do this:
pip install -U --force-reinstall charset-normalizer

pip install everything below:
python-dotenv==1.0.1
pydot==2.0.0
langchain==0.2.3
langchain-cohere==0.1.5
langchain-community==0.0.38
langchain-core==0.2.5
langchain-groq==0.1.3
langchain-text-splitters==0.2.1
langcodes==3.4.0
langdetect==1.0.9
langsmith==0.1.75
sentence-transformers==2.7.0
transformers==4.41.2
accelerate==0.30.1
conda install -c pytorch faiss-gpu

Now you will be able to run the batch file.
