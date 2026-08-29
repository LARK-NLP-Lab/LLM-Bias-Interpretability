import argparse
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_CHOICES = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

MAX_LENGTH = 128
N_SAMPLES_PER_RACE = 500

# Update these neuron lists to the final paper selections you want to analyze.
RACE_NEURONS = {
    "asian": [
        (27, 13406),
        (27, 5983),
        (26, 8641),
        (26, 6943),
        (26, 217),
        (26, 668),
        (24, 15029),
    ],
    "black": [
        (25, 7031),
        (25, 13507),
        (25, 1503),
    ],
    "latino": [
        (27, 4781),
        (27, 9876),
        (26, 18125),
        (26, 6242),
        (26, 7385),
        (24, 1321),
    ],
    "native_american": [
        (26, 6082),
        (24, 7087),
    ],
}

TOXIGEN_GROUPS = {
    "asian": ["asian", "chinese"],
    "black": ["black"],
    "latino": ["latino", "mexican"],
    "native_american": ["native_american"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hook-based per-neuron activation extraction for Qwen, Llama, and Mistral."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CHOICES.keys()),
        required=True,
        help="Model alias to analyze.",
    )
    parser.add_argument(
        "--samples-per-race",
        type=int,
        default=N_SAMPLES_PER_RACE,
        help="Maximum number of ToxiGen samples per race group.",
    )
    return parser.parse_args()


def resolve_model_name(model_choice):
    return MODEL_CHOICES[model_choice]


def model_setup(model_name):
    print(f"\nLoading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")
    print(f"Model type: {model.config.model_type}")
    return model, tokenizer


def load_toxigen_by_race(n_samples_per_race):
    print("\nLoading ToxiGen dataset...")
    dataset = load_dataset("toxigen/toxigen-data", "annotated", split="train")

    race_texts = {race: [] for race in TOXIGEN_GROUPS.keys()}

    for entry in dataset:
        target_group = entry["target_group"]
        text = entry["text"]

        if not text or len(text.strip()) == 0:
            continue

        for race_name, target_labels in TOXIGEN_GROUPS.items():
            if target_group in target_labels:
                race_texts[race_name].append(text)
                break

    rng = np.random.default_rng(42)
    for race_name in race_texts:
        if len(race_texts[race_name]) > n_samples_per_race:
            indices = rng.choice(len(race_texts[race_name]), n_samples_per_race, replace=False)
            race_texts[race_name] = [race_texts[race_name][i] for i in indices]

    print("\nTexts loaded per race group:")
    for race_name, texts in race_texts.items():
        print(f"  {race_name}: {len(texts)} texts")

    return race_texts


def get_mlp_down_proj_module(model, layer_idx):
    return model.model.layers[layer_idx].mlp.down_proj


class DownProjInputCapture:
    def __init__(self, model, layers_to_capture):
        self.model = model
        self.layers_to_capture = sorted(set(layers_to_capture))
        self.handles = []
        self.cache = {}

    def _make_hook(self, layer_idx):
        def hook(_module, module_input, _module_output):
            # module_input[0] is the pre-down-projection MLP activation with shape [batch, seq, d_mlp]
            self.cache[layer_idx] = module_input[0].detach()

        return hook

    def __enter__(self):
        for layer_idx in self.layers_to_capture:
            module = get_mlp_down_proj_module(self.model, layer_idx)
            handle = module.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self):
        self.cache.clear()


def extract_neuron_activations(model, tokenizer, texts, neurons_to_track):
    neurons_by_layer = defaultdict(list)
    for layer_idx, neuron_idx in neurons_to_track:
        neurons_by_layer[layer_idx].append(neuron_idx)

    neuron_activations = {neuron: [] for neuron in neurons_to_track}

    with DownProjInputCapture(model, neurons_by_layer.keys()) as capture:
        for text in tqdm(texts, desc="Extracting activations", leave=False):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
            ).to(model.device)

            capture.clear()
            with torch.no_grad():
                model(**inputs)

            for layer_idx, neuron_indices in neurons_by_layer.items():
                layer_activations = capture.cache[layer_idx]
                for neuron_idx in neuron_indices:
                    avg_act = layer_activations[0, :, neuron_idx].mean().item()
                    neuron_activations[(layer_idx, neuron_idx)].append(avg_act)

            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return neuron_activations


def compare_cross_race_activations(model, tokenizer, race_texts, race_neurons):
    print("\n" + "=" * 80)
    print("    NEURON ACTIVATION COMPARISON")
    print("=" * 80)

    all_results = {}

    for race_name, neurons in race_neurons.items():
        print(f"\n{'=' * 80}")
        print(f"   Analyzing {race_name.upper()} neurons")
        print(f"{'=' * 80}")

        race_results = {}

        for text_race_name, texts in race_texts.items():
            print(f"\n  Extracting activations on {text_race_name} texts...")
            activations = extract_neuron_activations(model, tokenizer, texts, neurons)
            race_results[text_race_name] = activations

        print(f"\n{'=' * 80}")
        print(f"   RESULTS FOR {race_name.upper()} NEURONS")
        print(f"{'=' * 80}")

        for layer_idx, neuron_idx in neurons:
            print(f"\n--- Layer {layer_idx + 1}, Neuron {neuron_idx} ---")
            print("  Per-race activations:")
            for text_race_name in race_texts.keys():
                acts = np.array(race_results[text_race_name][(layer_idx, neuron_idx)])
                marker = "←" if text_race_name == race_name else " "
                print(f"    {text_race_name:20s}: {np.mean(acts):+.6f} {marker}")

        all_results[race_name] = race_results

    return all_results


def main():
    args = parse_args()
    model_name = resolve_model_name(args.model)

    print("=" * 80)
    print("   CROSS-RACE NEURON ACTIVATION ANALYSIS")
    print("=" * 80)
    print(f"Model alias: {args.model}")
    print(f"Resolved model: {model_name}")

    model, tokenizer = model_setup(model_name)
    race_texts = load_toxigen_by_race(args.samples_per_race)
    compare_cross_race_activations(model, tokenizer, race_texts, RACE_NEURONS)


if __name__ == "__main__":
    main()
