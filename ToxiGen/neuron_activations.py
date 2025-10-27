import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from modeling_qwen2_mlp import Qwen2Config, Qwen2ForCausalLM

AutoModelForCausalLM.register(Qwen2Config, Qwen2ForCausalLM, exist_ok=True)

MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
MAX_LENGTH = 128
N_SAMPLES_PER_RACE = 500 

# Define race neurons (layer_idx is 0-indexed: actual layer 28 = layer 27 in code)
RACE_NEURONS = {
    'asian': [
        (27, 13406),  
        (27, 5983),  
        (26, 8641), 
        (26, 6943),
        (26, 217),
        (26, 668), 
        (24, 15029),  
    ],
    'black': [
        (25, 7031),
        (25, 13507),
        (25, 1503),
    ],
    'latino': [
        (27, 4781),   
        (27, 9876),  
        (26, 18125),
        (26, 6242),
        (26, 7385),  
        (24, 1321),   
    ],
    'native_american': [
        (26, 6082),
        (24, 7087),   
    ],
    'middle_eastern': [
        (27, 10528),
        (27, 9988), 
        (26, 9840),  
        (26, 8005), 
        (26, 6128),
        (26, 52), 
        (25, 13473),
        (25, 3012), 
        (24, 2602),  
    ]
}

TOXIGEN_GROUPS = {
    'asian': ['asian', 'chinese'],
    'black': ['black'],  
    'latino': ['latino', 'mexican'],
    'native_american': ['native_american'],
    'middle_eastern': ['middle_east', 'jewish', 'muslim']
}



def model_setup(model_name):
    """Loads model and tokenizer using the registered custom code."""
    print(f"\nLoading model: {model_name}...")
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
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")
    return model, tokenizer


def load_toxigen_by_race(n_samples_per_race):
    print("\nLoading ToxiGen dataset...")
    dataset = load_dataset("toxigen/toxigen-data", "annotated", split="train")
    
    race_texts = {race: [] for race in TOXIGEN_GROUPS.keys()}
    
    for entry in dataset:
        target_group = entry['target_group']
        text = entry['text']
    
        if not text or len(text.strip()) == 0:
            continue
        
        for race_name, target_labels in TOXIGEN_GROUPS.items():
            if target_group in target_labels:
                race_texts[race_name].append(text)
                break
    
    for race_name in race_texts:
        if len(race_texts[race_name]) > n_samples_per_race:
            np.random.seed(42)
            indices = np.random.choice(len(race_texts[race_name]), n_samples_per_race, replace=False)
            race_texts[race_name] = [race_texts[race_name][i] for i in indices]
    
    print("\nTexts loaded per race group:")
    for race_name, texts in race_texts.items():
        print(f"  {race_name}: {len(texts)} texts")
    
    return race_texts


def extract_neuron_activations(model, tokenizer, texts, neurons_to_track):
    """
    Args:
        neurons_to_track: list of (layer_idx, neuron_idx) tuples
        
    Returns:
        dict: {(layer_idx, neuron_idx): [activations]}
    """
    neurons_by_layer = {}
    for layer_idx, neuron_idx in neurons_to_track:
        if layer_idx not in neurons_by_layer:
            neurons_by_layer[layer_idx] = []
        neurons_by_layer[layer_idx].append(neuron_idx)
    
    neuron_activations = {neuron: [] for neuron in neurons_to_track}
    
    for text in tqdm(texts, desc="Extracting activations", leave=False):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_mlp_pre_residual=True)
        
        # extract MLP activations
        all_mlp_activations = outputs.mlp_outputs
        
        for layer_idx, neuron_indices in neurons_by_layer.items():
            layer_activations = all_mlp_activations[layer_idx]
            for neuron_idx in neuron_indices:
                avg_act = layer_activations[0, :, neuron_idx].mean().item()
                neuron_activations[(layer_idx, neuron_idx)].append(avg_act)
        
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return neuron_activations

## Compare how race-specific neurons respond to their own race vs other races
def compare_cross_race_activations(model, tokenizer, race_texts, race_neurons):
    print("\n" + "="*80)
    print("    NEURON ACTIVATION COMPARISON")
    print("="*80)
    
    all_results = {}
    
    for race_name, neurons in race_neurons.items():
        print(f"\n{'='*80}")
        print(f"   Analyzing {race_name.upper()} neurons")
        print(f"{'='*80}")
        
        race_results = {}
        
        for text_race_name, texts in race_texts.items():
            print(f"\n  Extracting activations on {text_race_name} texts...")
            activations = extract_neuron_activations(model, tokenizer, texts, neurons)
            race_results[text_race_name] = activations
        
        print(f"\n{'='*80}")
        print(f"   RESULTS FOR {race_name.upper()} NEURONS")
        print(f"{'='*80}")
        
        for layer_idx, neuron_idx in neurons:
            print(f"\n--- Layer {layer_idx+1}, Neuron {neuron_idx} ---")
            
            print(f"  Per-race activations:")
            for text_race_name in race_texts.keys():
                acts = np.array(race_results[text_race_name][(layer_idx, neuron_idx)])
                marker = "←" if text_race_name == race_name else " "
                print(f"    {text_race_name:20s}: {np.mean(acts):+.6f} {marker}")
        
        all_results[race_name] = race_results
    
    return all_results


def main():
    print("="*80)
    print("   CROSS-RACE NEURON ACTIVATION ANALYSIS")
    print("="*80)
    
    model, tokenizer = model_setup(MODEL_NAME)
    
    race_texts = load_toxigen_by_race(N_SAMPLES_PER_RACE)
    
    results = compare_cross_race_activations(model, tokenizer, race_texts, RACE_NEURONS)



if __name__ == "__main__":
    main()