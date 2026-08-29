import argparse
import os
import pickle

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_CHOICES = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

TOP_K_NEURONS = 20
TOP_K_TOKENS_PER_NEURON = 20
FINAL_N_LAYERS = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect top probe-aligned neurons in the final four MLP layers."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CHOICES.keys()),
        required=True,
        help="Model alias used for both model loading and probe matching.",
    )
    parser.add_argument(
        "--probe-path",
        required=True,
        help="Path to a saved probe .pkl file.",
    )
    parser.add_argument(
        "--top-k-neurons",
        type=int,
        default=TOP_K_NEURONS,
        help="Number of neurons to inspect per group per layer.",
    )
    parser.add_argument(
        "--top-k-tokens",
        type=int,
        default=TOP_K_TOKENS_PER_NEURON,
        help="Number of top projected tokens to show per neuron.",
    )
    return parser.parse_args()


def resolve_model_name(model_choice):
    return MODEL_CHOICES[model_choice]


def load_probe_file(probe_path):
    with open(probe_path, "rb") as file_obj:
        probe_data = pickle.load(file_obj)
    return probe_data


def resolve_probe_path(explicit_probe_path):
    if not os.path.exists(explicit_probe_path):
        raise FileNotFoundError(f"Probe file not found: {explicit_probe_path}")
    return explicit_probe_path


def load_probe_and_metadata(probe_path):
    print(f"Loading probe from: {probe_path}")

    probe_data = load_probe_file(probe_path)
    probe = probe_data["probe"]
    label_to_group = probe_data["label_to_group"]
    metadata = {key: value for key, value in probe_data.items() if key != "probe"}

    print("\nProbe Metadata:")
    print(f"  Model: {metadata.get('model_name')}")
    print(f"  Original probe layer: {metadata.get('probe_layer')}")
    print(f"  Test accuracy: {metadata.get('test_accuracy'):.4f}")
    print(f"  Race groups: {list(label_to_group.values())}")

    return probe, label_to_group, metadata


def model_setup(model_name):
    print(f"\nLoading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded. Number of layers: {model.config.num_hidden_layers}")
    return model, tokenizer


def get_final_layers(model, num_layers=FINAL_N_LAYERS):
    total_layers = model.config.num_hidden_layers
    start_idx = max(0, total_layers - num_layers)
    return list(range(start_idx, total_layers))


def find_neurons_for_race_in_layer(
    probe,
    model,
    tokenizer,
    layer_idx,
    class_idx,
    top_k_neurons,
    top_k_tokens,
):
    logit_probe = probe.named_steps["logisticregression"]
    W_race = logit_probe.coef_[class_idx]

    V_matrix = model.model.layers[layer_idx].mlp.down_proj.weight.data
    unembedding_matrix = model.get_output_embeddings().weight.data
    final_norm = model.model.norm

    W_race_tensor = torch.tensor(W_race, dtype=V_matrix.dtype, device=V_matrix.device)
    W_race_normalized = W_race_tensor / torch.norm(W_race_tensor)

    similarities = []
    for neuron_idx in range(V_matrix.shape[1]):
        v_i = V_matrix[:, neuron_idx]
        v_i_normalized = v_i / torch.norm(v_i)
        cos_sim = torch.dot(W_race_normalized, v_i_normalized).item()
        similarities.append(cos_sim)

    similarities = np.array(similarities)
    top_neuron_indices = np.argsort(similarities)[-top_k_neurons:][::-1]

    neuron_results = []
    for rank, neuron_idx in enumerate(top_neuron_indices, start=1):
        v_i = V_matrix[:, neuron_idx]
        v_i_normed = final_norm(v_i)
        logit_lens_scores = unembedding_matrix @ v_i_normed

        top_tokens = torch.topk(logit_lens_scores, top_k_tokens)
        decoded_tokens = [tokenizer.decode([token_id]) for token_id in top_tokens.indices]
        token_scores = top_tokens.values.detach().cpu().numpy()

        neuron_results.append(
            {
                "rank": rank,
                "neuron_idx": int(neuron_idx),
                "cosine_similarity": float(similarities[neuron_idx]),
                "top_tokens": decoded_tokens,
                "token_scores": token_scores.tolist(),
            }
        )

    return neuron_results


def analyze_final_layers_and_groups(
    probe,
    model,
    tokenizer,
    label_to_group,
    top_k_neurons,
    top_k_tokens,
):
    print("\n" + "=" * 80)
    print("   ANALYZING RACE NEURONS IN THE FINAL FOUR MLP LAYERS")
    print("=" * 80)

    layers_to_analyze = get_final_layers(model)
    print(f"Analyzing layers (0-indexed): {layers_to_analyze}")

    all_results = {}

    for layer_idx in layers_to_analyze:
        display_layer = layer_idx + 1

        print(f"\n{'=' * 80}")
        print(f"   MLP LAYER {display_layer} (0-indexed: {layer_idx})")
        print(f"{'=' * 80}")

        layer_results = {}

        for class_idx, group_name in label_to_group.items():
            neuron_results = find_neurons_for_race_in_layer(
                probe,
                model,
                tokenizer,
                layer_idx,
                class_idx,
                top_k_neurons,
                top_k_tokens,
            )

            print(f"\n--- Top {top_k_neurons} Neurons for '{group_name}' ---")
            for neuron_info in neuron_results:
                tokens_str = ", ".join([f"'{token}'" for token in neuron_info["top_tokens"]])
                print(
                    f"  - Neuron {neuron_info['neuron_idx']} "
                    f"(Cosine Sim: {neuron_info['cosine_similarity']:.4f})"
                )
                print(f"    Top Tokens: [{tokens_str}]")

            layer_results[group_name] = neuron_results

        all_results[layer_idx] = layer_results

    return all_results


def main():
    args = parse_args()
    model_name = resolve_model_name(args.model)
    probe_path = resolve_probe_path(args.probe_path)

    print("=" * 80)
    print("   RACE NEURON FINDER")
    print("=" * 80)
    print(f"Model alias: {args.model}")
    print(f"Resolved model: {model_name}")

    probe, label_to_group, _metadata = load_probe_and_metadata(probe_path)
    model, tokenizer = model_setup(model_name)

    analyze_final_layers_and_groups(
        probe,
        model,
        tokenizer,
        label_to_group,
        args.top_k_neurons,
        args.top_k_tokens,
    )

    print("\n" + "=" * 80)
    print("   ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
