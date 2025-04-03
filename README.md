**EmotioNotes** is a curated dataset of historical concert program notes annotated with emotional content, designed to support research in music emotion recognition (MER), music information retrieval, and cultural musicology.

This dataset leverages archival materials from the **New York Philharmonic** dating back to the 1840s. Each entry includes metadata about the performance, detailed program notes, and estimated emotional scores in **valence**, **arousal**, and **dominance** (VAD) dimensions.

---

## 📦 Dataset Description

Each JSON entry in the dataset corresponds to a specific performance of a classical music work and includes the following fields:

### 🎼 Performance Metadata
- `id`: Unique identifier
- `programID`: Identifier of the original concert program
- `orchestra`: Performing ensemble
- `season`: Concert season (e.g., "1842–43")
- `concerts`: List of performance details:
  - `eventType`: e.g., "Subscription Season", "Special"
  - `location`: e.g., "Manhattan, NY"
  - `venue`: Venue name
  - `date`: Date of performance
  - `time`: Time of performance

### 🎶 Musical Work Details
- `workTitle`: Title of the performed piece
- `movement`: Specific movement (if available)
- `composerName`: Name of the composer
- `conductorName`: Name of the conductor
- `soloists`: List of soloists with:
  - `soloistName`
  - `soloistInstrument`
  - `soloistRoles`

### 📝 Program Note
- `ProgramNote`: Extracted textual description of the musical work, drawn from archival concert booklets.

### Emotion Annotations
Emotion scores are derived using a lexicon-based method (Warriner et al., 2013):
- `valence_mean`: Pleasantness of a stimulus (0–1 scale)
- `arousal_mean`: Intensity/energy (0–1 scale)
- `dominance_mean`: Sense of control (0–1 scale)
- `valence_std`, `arousal_std`, `dominance_std`: Standard deviation of emotional scores

### Metrics for avoiding hallucinations
- `BLEUScore`: A text similarity score used to evaluate the quality of extracted program notes against the raw OCR text.

---

## 📊 Use Cases

This dataset is ideal for:
- Text-based emotion recognition in music
- Music recommendation systems based on emotion
- Cultural analysis of how music was described across centuries
- Using text-based emotion annotations to audio/music scores as proxies

---

## 📁 Format

The dataset is provided in **JSON format**. Example entry:

```json
{
  "workTitle": "Symphony No. 3 In E Flat Major, Op. 55 (Eroica)",
  "composerName": "Ludwig van Beethoven",
  "ProgramNote": "This great work was commenced when Napoleon was first Consul...",
  "valence_mean": 0.585,
  "arousal_mean": 0.384,
  "dominance_mean": 0.618
}
```

---

Install miniconda if you don't have it:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

chmod a+x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh

Note - While going through CondaÃ¢â‚¬â„¢s guided installer, it will ask you to specify an installation directory. This will default to your home directory - which has a 14gb disk restriction as a student. You will likely want to  change this path to use the SoundBendOR Lab filespace. /to/soundbendor/<ONID>

Create a new conda environment and activate it
conda create -n program_notes python=3.12

conda activate program_notes


pip install jupyterlab

# Load these modules
module load gcc cmake cuda/11.8 slurm

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

If you get this error and canÃ¢â‚¬â„¢t open Jupyter Session:
AttributeError: partially initialized module 'charset_normalizer' has no attribute 'md__mypyc' (most likely due to a circular import)
Do this:
pip install -U --force-reinstall charset-normalizer

#pip install everything below:
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
