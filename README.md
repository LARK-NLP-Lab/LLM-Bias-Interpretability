# LLM-Bias-Interpretability

Code for the paper **["Tracing the Latent Threads: A Mechanistic Study of How LLMs Represent and Operationalize Race and Ethnicity Cues."](https://arxiv.org/html/2601.12868v1)**

This repository studies how race and ethnicity cues are represented inside large language models using a pipeline of:
- linear probing
- neuron selection via cosine similarity to probe directions
- neuron activation analysis
- targeted neuron intervention

We evaluate three open-source instruction-tuned models:
- [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

We use two datasets:
- `ToxiGen` for toxicity-related text
- `C-REACT` for clinical race and ethnicity cues

## Repo Layout

- `ToxiGen/linear_probing.py`
  Train the ToxiGen probe and inspect probe directions.
- `ToxiGen/select_neurons.py`
  Rank candidate neurons from the final four MLP layers and print top projected tokens for manual selection.
- `ToxiGen/neuron_activations_hooked.py`
  Final ToxiGen activation analysis script. This is the script used for cross-model MLP neuron activation extraction.
- `C-REACT/linear_probe.py`
  Train the C-REACT probe and inspect probe directions.
- `C-REACT/select_neurons.py`
  Inspect candidate neurons in the final four MLP layers.
- `C-REACT/prediction_bias.py`
  Run indirect-cue classification, collect misclassified samples, and record neuron activations.
- `C-REACT/neuron_intervention.py`
  Run intervention experiments on selected neuron groups.

## Requirements

Install the main Python dependencies with:

```bash
pip install -r requirements.txt
```

## Data

- [`ToxiGen`](https://huggingface.co/datasets/toxigen/toxigen-data) is loaded directly from Hugging Face via `datasets`.
- `C-REACT` requires local copies of:
  - `indicators_df.jsonl`
  - `all_re_assignments_df.jsonl`

The clinical data is based on [`C-REACT`](https://physionet.org/content/race-ethnicity-clinical-text/1.0.0/), which is hosted on PhysioNet and requires approved access.

Place the C-REACT files in the `C-REACT/` directory before running the clinical experiments.

## Running The Pipeline

### ToxiGen

Train a probe:

```bash
python ToxiGen/linear_probing.py --model qwen
python ToxiGen/linear_probing.py --model llama
python ToxiGen/linear_probing.py --model mistral
```

Inspect candidate neurons from a saved probe:

```bash
python ToxiGen/select_neurons.py --model qwen --probe-path /path/to/probe.pkl
```

Run activation analysis:

```bash
python ToxiGen/neuron_activations_hooked.py --model qwen
```

### C-REACT

Train a probe:

```bash
python C-REACT/linear_probe.py --model qwen
python C-REACT/linear_probe.py --model llama
python C-REACT/linear_probe.py --model mistral
```

Inspect candidate neurons:

```bash
python C-REACT/select_neurons.py --model qwen --probe-path /path/to/probe.pkl
```

Run prediction bias analysis and activation extraction:

```bash
python C-REACT/prediction_bias.py --model qwen
```

Run intervention experiments:

```bash
python C-REACT/neuron_intervention.py --model qwen
```

## Notes

- `ToxiGen/neuron_activations_hooked.py` is the recommended activation-analysis script for the final version of this repository.
- Probe files are saved locally during training and are then reused for neuron selection.
- Some scripts contain manually curated neuron lists used in the paper experiments.
