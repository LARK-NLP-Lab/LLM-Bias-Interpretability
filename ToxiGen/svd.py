import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import numpy as np


MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
PROBE_PATH = "race_probe_results/race_probe_20251023_190806_28.pkl"  

ANALYSIS_LAYERS = [25, 26, 27] 
NUM_NEURONS_FOR_SVD = 128  # N: Number of top neurons to define the subspace (same as toxicity paper)
NUM_SVD_COMPONENTS = 5     # K: Number of top SVD components to decode
TOP_K_TOKENS_TO_DECODE = 10 



def model_setup(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=False
    )
    model.eval()
    print(f"Model loaded.")
    return model, tokenizer


def load_probe_and_metadata(probe_path):
    print(f"Loading probe from: {probe_path}")
    
    with open(probe_path, 'rb') as f:
        probe_data = pickle.load(f)
    
    probe = probe_data['probe']
    label_to_group = probe_data['label_to_group']
    
    print(f"  Race groups: {list(label_to_group.values())}")
    
    return probe, label_to_group


def svd_analysis(probe, model, tokenizer, layer_idx, num_top_neurons, num_svd_components, label_to_group):
    print("\n" + "="*80)
    print(f"   SVD ANALYSIS - MLP Layer {layer_idx}")
    print("="*80)

    logit_probe = probe.named_steps['logisticregression']
    mlp = model.model.layers[layer_idx].mlp
    final_norm = model.model.norm
    unembedding = model.get_output_embeddings().weight

    V_matrix_original_T = mlp.down_proj.weight.T.data

    V_matrix_normed = final_norm(V_matrix_original_T.to(torch.float32)).to(V_matrix_original_T.dtype)

    for class_idx in range(len(logit_probe.classes_)):
        race_class = label_to_group[class_idx] 
        print(f"\n{'='*80}")
        print(f"   Race Group: '{race_class.upper()}'")
        print(f"{'='*80}")

        # Get probe weights for each race
        W_race_numpy = logit_probe.coef_[class_idx]
        W_race_tensor = torch.tensor(W_race_numpy, dtype=V_matrix_normed.dtype, device=V_matrix_normed.device)

        # 1. Find Top N Neurons by dot product
        print(f"\nStep 1: Finding top {num_top_neurons} neurons")
        dot_products = torch.einsum("nd,d->n", V_matrix_normed, W_race_tensor)
        _, top_neuron_indices = torch.topk(dot_products, num_top_neurons)
        print(f"  Identified top {num_top_neurons} neurons")

        # 2. Stack value vectors for SVD (shape: (num_top_neurons, hidden_dim))
        print(f"\nStep 2: Stacking value vectors for SVD")
        value_vectors_for_svd = V_matrix_original_T[top_neuron_indices].to(torch.float32)
        print(f"  Matrix shape for SVD: {value_vectors_for_svd.shape}")

        # 3. Perform SVD
        print(f"\nStep 3: Performing SVD")
        try:
            # U: (num_top_neurons, num_top_neurons)
            # S: (min(num_top_neurons, hidden_dim),)
            # Vh: (hidden_dim, hidden_dim)
            U, S, Vh = torch.linalg.svd(value_vectors_for_svd, full_matrices=False)
            print(f"  SVD done")
            print(f"  Singular values (top 5): {S[:5].cpu().numpy()}")
        except torch.linalg.LinAlgError as e:
            print(f"  SVD failed for '{race_class}': {e}")
            continue

        # 4. Decode and display the top K SVD components
        print(f"\nStep 4: Decoding top {num_svd_components} SVD components via LogitLens")
        print(f"\n--- Top {num_svd_components} SVD Components for '{race_class}' ---")
        
        for k in range(min(num_svd_components, Vh.shape[0])):
            component_vector = Vh[k, :].to(unembedding.dtype).to(unembedding.device)

            component_normed = final_norm(component_vector)
            logit_lens_scores = unembedding @ component_normed
            
            top_tokens = torch.topk(logit_lens_scores, TOP_K_TOKENS_TO_DECODE)
            decoded_tokens = [tokenizer.decode([tok_id]) for tok_id in top_tokens.indices]

            print(f"\n  SVD Component {k} (Singular value: {S[k].item():.4f}):")
            print(f"    Top {TOP_K_TOKENS_TO_DECODE} tokens: {decoded_tokens}")

    print("\n" + "="*80)


def main():
    print("="*80)
    print("   SVD ANALYSIS FOR RACE NEURONS")
    print("="*80)
    
    model, tokenizer = model_setup(MODEL_NAME)
    
    probe, label_to_group = load_probe_and_metadata(PROBE_PATH)
    
    for layer_idx in ANALYSIS_LAYERS:
        print(f"\n\n{'#'*80}")
        print(f"   ANALYZING LAYER {layer_idx+1})")
        print(f"{'#'*80}")
        
        svd_analysis(
            probe,
            model,
            tokenizer,
            layer_idx=layer_idx,
            num_top_neurons=NUM_NEURONS_FOR_SVD,
            num_svd_components=NUM_SVD_COMPONENTS,
            label_to_group=label_to_group  
        )
    
    print("\n" + "="*80)
    print("   ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()