**EmotioNotes** is a curated dataset of historical concert program notes annotated with emotional content, designed to support research in music emotion recognition (MER), music information retrieval, and cultural musicology.

This dataset leverages archival materials from the **New York Philharmonic** dating back to the 1840s. Each entry includes metadata about the performance, detailed program notes, and estimated emotional scores in **valence**, **arousal**, and **dominance** (VAD) dimensions.


**Explore it online:** [EmotioNotes Explorer](https://soundbendor-emotionotes.streamlit.app/), where you can search the notes by composer or piece and see each note's emotion words highlighted.

## ⬇️ Download

The data files are attached to the [**v1.0 release**](https://github.com/Soundbendor/program_notes_data/releases/tag/v1.0) and are not stored in the git tree.

| File | Contents |
|---|---|
| [`emotionotes_v1.json.gz`](https://github.com/Soundbendor/program_notes_data/releases/download/v1.0/emotionotes_v1.json.gz) | **The EmotioNotes dataset** described in the paper: 31,966 program notes from 9,569 programs (1842–2023), with performance metadata, VAD scores and `BLEUScore`. gzip-compressed JSON list. |
| [`emotionotes_v1.parquet`](https://github.com/Soundbendor/program_notes_data/releases/download/v1.0/emotionotes_v1.parquet) | The same 31,966 notes as Parquet, with cleaned columns (`composer`, `work`, `note`, `valence`, `arousal`, `dominance`, `bleu`, parsed `date` and `decade`). This is the file used by the online explorer. |
| [`emotionotes_v1_before_bleu_filter.json.gz`](https://github.com/Soundbendor/program_notes_data/releases/download/v1.0/emotionotes_v1_before_bleu_filter.json.gz) | All 46,992 extracted and scored notes **before** the BLEU hallucination filter. |
| [`nyphilharmonic_programs.json.gz`](https://github.com/Soundbendor/program_notes_data/releases/download/v1.0/nyphilharmonic_programs.json.gz) | The OCR'd New York Philharmonic program text used as extraction input. |
| [`SHA256SUMS`](https://github.com/Soundbendor/program_notes_data/releases/download/v1.0/SHA256SUMS) | Checksums; verify with `sha256sum -c SHA256SUMS`. |

```python
import gzip, json
notes = json.load(gzip.open("emotionotes_v1.json.gz", "rt"))
```

**Sources and rights.** Performance metadata comes from the NY Philharmonic [Performance History](https://github.com/nyphilarchive/PerformanceHistory) (CC0). Program-note text comes from the NY Philharmonic Shelby White & Leon Levy [Digital Archives](https://archives.nyphil.org/), which publishes no explicit reuse licence; it is shared here for non-commercial research and teaching, with attribution to the New York Philharmonic Archives.

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

## 📚 Citation

If you use this dataset, please cite the following paper:

> Khanal, P., & Donnelly, P. J. (2025). *EmotioNotes Dataset: Decoding Emotions in Classical Music Through Concert Program Notes*. In: Machado, P., Johnson, C., Santos, I. (eds) Artificial Intelligence in Music, Sound, Art and Design. EvoMUSART 2025. Lecture Notes in Computer Science, vol 15611, pp. 339–356. Springer, Cham. https://doi.org/10.1007/978-3-031-90167-6_23

```bibtex
@inproceedings{khanal2025emotionotes,
  author    = {Khanal, Pratik and Donnelly, Patrick J.},
  title     = {EmotioNotes Dataset: Decoding Emotions in Classical Music Through Concert Program Notes},
  booktitle = {Artificial Intelligence in Music, Sound, Art and Design (EvoMUSART 2025)},
  series    = {Lecture Notes in Computer Science}, volume = {15611}, pages = {339--356},
  publisher = {Springer}, year = {2025}, doi = {10.1007/978-3-031-90167-6_23}
}
```

📂 [Dataset & Code](https://github.com/Soundbendor/program_notes_data) · 🔎 [Online explorer](https://soundbendor-emotionotes.streamlit.app/)

## 🙌 Acknowledgments

This project would not have been possible without the following:

- The **New York Philharmonic Digital Archives**, for access to centuries of concert program materials.
- **Meta AI** for open-sourcing **LLaMA 3.1**, which enabled structured text extraction from unstructured OCR text.
- **Warriner et al. (2013)** for the valence, arousal, and dominance (VAD) emotion lexicon used for annotation.

## Getting Started

Install miniconda if you don't have it:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

chmod a+x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh

Note - While going through Conda's guided installer, it will ask you to specify an installation directory. This will default to your home directory - which has a 14gb disk restriction as a student. You will likely want to  change this path to use the SoundBendOR Lab filespace. /to/soundbendor/<ONID>

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

The scripts read `HF_TOKEN` (a Hugging Face access token with Llama 3.1 access) and, for the RAG variants, `COHERE_API_KEY` from a `.env` file in this folder. `.env` is git-ignored; never commit it.
