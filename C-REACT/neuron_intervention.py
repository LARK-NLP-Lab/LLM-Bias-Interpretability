import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import collections
import os

MODEL_NAME = 'meta-llama/Llama-3.1-8B-Instruct'
# MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.3'
# MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
MAX_LENGTH = 512

# Input file with misclassified samples
MISCLASSIFIED_FILE = 'misclassified_samples_with_activations_llama3_new.jsonl'

TARGET_RACES = {
    'White': 'White',
    'Black/AA': 'Black or African American',
    'Asian': 'Asian'
}

# Neurons to intervene on
ALL_RACE_NEURONS = {
    'Asian': {
        'Direct': [
            (31, 5691), (30, 14299), (28, 5272),
        ],
        'Indirect': [
            (31, 5691), (31, 6950), (28, 10616),
        ]
    },
    'Black/AA': {
        'Direct': [
            (29, 7195), (29, 3868), (28, 13826), (28, 6824),
        ],
        'Indirect': [
            (29, 7195), (28, 6824), (28, 13826),
        ]
    },
    'White': {
        'Direct': [
            (30, 9094), 
        ],
        'Indirect': [
            (31, 10606), (30, 12584), (31,10409),
            (28, 4193),
        ]
    }
}

def model_setup(model_name):
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
    print(f"Intermediate size: {model.config.intermediate_size}")
    
    model_type = model.config.model_type
    print(f"Model type: {model_type}")
    
    return model, tokenizer, model_type


def load_misclassified_samples(filepath, filter_ground_truth=None, filter_predicted=None):
    """
    Load misclassified samples from JSONL file
    """
    samples = []
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            
            if filter_ground_truth and sample['ground_truth'] != filter_ground_truth:
                continue
            if filter_predicted and sample['predicted_race'] != filter_predicted:
                continue
                
            samples.append(sample)
    
    print(f"Loaded {len(samples)} samples", end="")
    if filter_ground_truth or filter_predicted:
        print(f" (filtered: ground_truth={filter_ground_truth}, predicted={filter_predicted})")
    else:
        print()
    
    return samples


def get_mlp_module(model, layer_idx, model_type):
    """Returns the MLP down_proj module for a given layer."""
    if model_type in ['llama', 'mistral', 'qwen2']:
        return model.model.layers[layer_idx].mlp.down_proj
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def generate_with_intervention(model, tokenizer, text, neurons_to_intervene, model_type, amplification_factor):
    prompt_options = ', '.join(TARGET_RACES.values())
    prompt = (
        "Please read the following sentence from a clinical note. "
        "Based on the information contained within the sentence, determine which of the following races or ethnicities the patient belongs to. "
        "Please respond with only one option.\n\n"
        f"Options: [{prompt_options}]\n\n"
        f"Clinical Note: \"{text}\"\n\n"
        "Inferred Race or Ethnicity:"
    )

    hooks = []
    
    # Create intervention hooks
    def make_intervention_hook(layer_idx, neuron_indices, amp_factor):
        def hook(module, input, output):
            modified_input = input[0].clone()

            for neuron_idx in neuron_indices:
                activation = modified_input[:, :, neuron_idx]

                modified_input[:, :, neuron_idx] = torch.where(
                    activation < 0,
                    activation * amp_factor,      # negative -> more negative
                    activation * -amp_factor      # positive -> flip to negative
                )

            # Recompute output with modified input
            new_output = torch.nn.functional.linear(modified_input, module.weight, module.bias)
            return new_output
        return hook
    
    # Group neurons by layer
    neurons_by_layer = collections.defaultdict(list)
    for layer_idx, neuron_idx in neurons_to_intervene:
        neurons_by_layer[layer_idx].append(neuron_idx)
    
    # Register hooks
    for layer_idx, neuron_indices in neurons_by_layer.items():
        mlp_module = get_mlp_module(model, layer_idx, model_type)
        hook = mlp_module.register_forward_hook(
            make_intervention_hook(layer_idx, neuron_indices, amplification_factor)
        )
        hooks.append(hook)
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        output_tokens = outputs[0, inputs.input_ids.shape[-1]:]
        raw_output = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        
        # Extract prediction
        predicted_race_key = None
        for race_key, race_label in TARGET_RACES.items():
            if race_label.lower() in raw_output.lower() or race_key.lower() in raw_output.lower():
                predicted_race_key = race_key
                break
        
    finally:
        for hook in hooks:
            hook.remove()
    
    return predicted_race_key, raw_output


def run_intervention_experiment(model, tokenizer, model_type, samples, neurons_to_intervene, experiment_name, amplification_factor):
    print(f"\n{'='*80}")
    print(f"   INTERVENTION EXPERIMENT: {experiment_name}")
    print(f"   Amplification Factor: {amplification_factor}")
    print(f"   Intervening on {len(neurons_to_intervene)} neurons")
    print(f"{'='*80}")

    change_counts = collections.defaultdict(int)

    for sample in tqdm(samples, desc=f"Running Factor {amplification_factor}"):
        text = sample['sentence_text']

        new_prediction, new_output = generate_with_intervention(
            model, tokenizer, text, neurons_to_intervene, model_type, amplification_factor
        )

        if new_prediction is None:
            change_counts['Unknown'] += 1
        else:
            change_counts[new_prediction] += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print results table
    total = len(samples)
    print(f"\n{'='*60}")
    print(f"   RESULTS: {experiment_name} | Factor: {amplification_factor}")
    print(f"{'='*60}")
    print(f"   Original Prediction: '{samples[0]['predicted_race']}'")
    print(f"   Ground Truth: '{samples[0]['ground_truth']}'")
    print(f"   Total samples: {total}")
    print(f"{'='*60}")
    print(f"   {'New Prediction':<20} | {'Count':<10} | {'Percentage':<10}")
    print(f"   {'-'*50}")
    
    for prediction in ['White', 'Black/AA', 'Asian', 'Unknown']:
        count = change_counts.get(prediction, 0)
        pct = 100 * count / total if total > 0 else 0
        
        marker = ""
        if prediction == samples[0]['ground_truth']:
            marker = " <-- Ground Truth"
        elif prediction == samples[0]['predicted_race']:
            marker = " <-- No Change"
            
        print(f"   {prediction:<20} | {count:<10} | {pct:>6.1f}%{marker}")
    
    print(f"{'='*60}")

    return change_counts


def main():
    print("="*80)
    print("   NEURON INTERVENTION EXPERIMENT")
    print("="*80)
    
    model, tokenizer, model_type = model_setup(MODEL_NAME)

    samples = load_misclassified_samples(
        MISCLASSIFIED_FILE,
        filter_ground_truth='White',
        filter_predicted='Asian'
    )
    
    if not samples:
        print("No samples found matching the filter criteria.")
        return

    target_neurons_direct = ALL_RACE_NEURONS['Asian']['Indirect']
    target_neurons_all = list(set(target_neurons_direct)) 
    
    print(f"\nNeurons to intervene on:")
    print(f"  Combined (unique): {target_neurons_all}")
    
    factors = [5, 10, 20]
    
    for factor in factors:
        run_intervention_experiment(
            model, tokenizer, model_type, samples,
            neurons_to_intervene=target_neurons_all,
            experiment_name="Intervention Experiment",
            amplification_factor=factor
        )

if __name__ == "__main__":
    main()


# import torch
# import numpy as np
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from tqdm import tqdm
# import json
# import collections
# import os

# # --- GLOBAL CONFIGURATION ---
# MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.3'
# MAX_LENGTH = 512

# # Input file with misclassified samples
# MISCLASSIFIED_FILE = 'misclassified_samples_with_activations_mistral_new.jsonl'

# # Define the target labels for the experiment
# TARGET_RACES = {
#     'White': 'White',
#     'Black/AA': 'Black or African American',
#     'Asian': 'Asian'
# }

# # --- NEURONS TO INTERVENE ON ---
# ALL_RACE_NEURONS = {
#     'Asian': {
#         'Direct': [
#             (31, 4453), (30, 2346), (29, 2137), (29, 8986),
#         ],
#     },
#     'Black/AA': {
#         'Direct': [
#             (31, 5923), (30, 8715), (29, 3398), (29, 12572), (28, 13186),
#         ],
#         'Indirect': [
#             (30, 8715),
#         ]
#     },
#     'White': {
#         'Direct': [
#             (31, 1606), (31, 12760), (31, 9831), 
#         ],
#         'Indirect': [
#             (31, 2399), (30, 7356), (29,4487),
#             (28, 260),
#         ]
#     }
# }


# def model_setup(model_name):
#     """Loads model and tokenizer with local caching."""
#     # Create a local cache folder
#     my_local_cache = "./my_model_cache"
#     os.makedirs(my_local_cache, exist_ok=True)
    
#     print(f"Loading tokenizer and model from local cache: {my_local_cache}")

#     # 1. Load Tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name,
#         cache_dir=my_local_cache 
#     )

#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     # 2. Load Model
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float32,
#         device_map="auto",
#         cache_dir=my_local_cache 
#     )
    
#     model.eval()
#     print(f"Model loaded. Layers: {model.config.num_hidden_layers}")
#     print(f"Intermediate size: {model.config.intermediate_size}")
    
#     model_type = model.config.model_type
#     print(f"Model type: {model_type}")
    
#     return model, tokenizer, model_type


# def load_misclassified_samples(filepath, filter_ground_truth=None, filter_predicted=None):
#     """
#     Load misclassified samples from JSONL file.
#     Optionally filter by ground_truth and/or predicted race.
#     """
#     samples = []
#     if not os.path.exists(filepath):
#         print(f"Error: File '{filepath}' not found.")
#         return []

#     with open(filepath, 'r', encoding='utf-8') as f:
#         for line in f:
#             sample = json.loads(line.strip())
            
#             # Apply filters if specified
#             if filter_ground_truth and sample['ground_truth'] != filter_ground_truth:
#                 continue
#             if filter_predicted and sample['predicted_race'] != filter_predicted:
#                 continue
                
#             samples.append(sample)
    
#     print(f"Loaded {len(samples)} samples", end="")
#     if filter_ground_truth or filter_predicted:
#         print(f" (filtered: ground_truth={filter_ground_truth}, predicted={filter_predicted})")
#     else:
#         print()
    
#     return samples


# def get_mlp_module(model, layer_idx, model_type):
#     """Returns the MLP down_proj module for a given layer."""
#     if model_type in ['llama', 'mistral', 'qwen2']:
#         return model.model.layers[layer_idx].mlp.down_proj
#     else:
#         raise ValueError(f"Unsupported model type: {model_type}")


# def generate_with_intervention(model, tokenizer, text, neurons_to_intervene, model_type, amplification_factor):
#     """
#     Generate model output while amplifying/flipping specific neurons.
#     """
#     prompt_options = ', '.join(TARGET_RACES.values())
#     prompt = (
#         "Please read the following sentence from a clinical note. "
#         "Based on the information contained within the sentence, determine which of the following races or ethnicities the patient belongs to. "
#         "Please respond with only one option.\n\n"
#         f"Options: [{prompt_options}]\n\n"
#         f"Clinical Note: \"{text}\"\n\n"
#         "Inferred Race or Ethnicity:"
#     )

#     hooks = []
    
#     # Create intervention hooks
#     def make_intervention_hook(layer_idx, neuron_indices, amp_factor):
#         def hook(module, input, output):
#             modified_input = input[0].clone()

#             for neuron_idx in neuron_indices:
#                 activation = modified_input[:, :, neuron_idx]

#                 # Flip and amplify:
#                 # If activation is negative, make it more negative (amplify feature presence if feature is negative)
#                 # If activation is positive, flip to negative (suppress feature if feature is positive)
#                 modified_input[:, :, neuron_idx] = torch.where(
#                     activation < 0,
#                     activation * amp_factor,      
#                     activation * -amp_factor      
#                 )

#             # Recompute output with modified input
#             new_output = torch.nn.functional.linear(modified_input, module.weight, module.bias)
#             return new_output
#         return hook
    
#     # Group neurons by layer
#     neurons_by_layer = collections.defaultdict(list)
#     for layer_idx, neuron_idx in neurons_to_intervene:
#         neurons_by_layer[layer_idx].append(neuron_idx)
    
#     # Register hooks
#     for layer_idx, neuron_indices in neurons_by_layer.items():
#         mlp_module = get_mlp_module(model, layer_idx, model_type)
#         # Pass the specific amplification_factor here
#         hook = mlp_module.register_forward_hook(
#             make_intervention_hook(layer_idx, neuron_indices, amplification_factor)
#         )
#         hooks.append(hook)
    
#     try:
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
        
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=10,
#                 do_sample=False,
#                 pad_token_id=tokenizer.eos_token_id
#             )
        
#         output_tokens = outputs[0, inputs.input_ids.shape[-1]:]
#         raw_output = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        
#         # Extract prediction
#         predicted_race_key = None
#         for race_key, race_label in TARGET_RACES.items():
#             if race_label.lower() in raw_output.lower() or race_key.lower() in raw_output.lower():
#                 predicted_race_key = race_key
#                 break
        
#     finally:
#         # Remove all hooks
#         for hook in hooks:
#             hook.remove()
    
#     return predicted_race_key, raw_output


# def run_intervention_experiment(model, tokenizer, model_type, samples, neurons_to_intervene, experiment_name, amplification_factor):
#     """
#     Run intervention experiment and print a results table for the specific amplification factor.
#     """
#     print(f"\n{'='*80}")
#     print(f"   INTERVENTION EXPERIMENT: {experiment_name}")
#     print(f"   Amplification Factor: {amplification_factor}")
#     print(f"   Intervening on {len(neurons_to_intervene)} neurons")
#     print(f"{'='*80}")

#     # Track changes: {new_prediction: count}
#     change_counts = collections.defaultdict(int)

#     for sample in tqdm(samples, desc=f"Running Factor {amplification_factor}"):
#         text = sample['sentence_text']
        
#         # Run with intervention
#         new_prediction, new_output = generate_with_intervention(
#             model, tokenizer, text, neurons_to_intervene, model_type, amplification_factor
#         )

#         if new_prediction is None:
#             change_counts['Unknown'] += 1
#         else:
#             change_counts[new_prediction] += 1

#         # Cleanup
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
    
#     # Print results table
#     total = len(samples)
#     print(f"\n{'='*60}")
#     print(f"   RESULTS: {experiment_name} | Factor: {amplification_factor}")
#     print(f"{'='*60}")
#     print(f"   Original Prediction: '{samples[0]['predicted_race']}'")
#     print(f"   Ground Truth: '{samples[0]['ground_truth']}'")
#     print(f"   Total samples: {total}")
#     print(f"{'='*60}")
#     print(f"   {'New Prediction':<20} | {'Count':<10} | {'Percentage':<10}")
#     print(f"   {'-'*50}")
    
#     for prediction in ['White', 'Black/AA', 'Asian', 'Unknown']:
#         count = change_counts.get(prediction, 0)
#         pct = 100 * count / total if total > 0 else 0
        
#         marker = ""
#         if prediction == samples[0]['ground_truth']:
#             marker = " <-- Ground Truth"
#         elif prediction == samples[0]['predicted_race']:
#             marker = " <-- No Change"
            
#         print(f"   {prediction:<20} | {count:<10} | {pct:>6.1f}%{marker}")
    
#     print(f"{'='*60}")
    
#     return change_counts


# def main():
#     print("="*80)
#     print("   NEURON INTERVENTION EXPERIMENT - MULTIPLE FACTORS")
#     print("="*80)

#     # 1. Load model
#     model, tokenizer, model_type = model_setup(MODEL_NAME)

#     # 2. Load misclassified samples (White -> Asian)
#     samples = load_misclassified_samples(
#         MISCLASSIFIED_FILE,
#         filter_ground_truth='White',
#         filter_predicted='Black/AA'
#     )

#     if not samples:
#         print("No samples found matching the filter criteria.")
#         return

#     # 3. Define neurons
#     # Intervening on Asian neurons to correct White -> Asian error
#     asian_neurons_direct = ALL_RACE_NEURONS['Black/AA']['Indirect']
#     asian_neurons_all = list(set(asian_neurons_direct))

#     print(f"\nNeurons to intervene on (Asian):")
#     print(f"  Combined (unique): {asian_neurons_all}")

#     # 4. Run experiments with different amplification factors
#     amplification_factors = [5, 10, 20]

#     for amp_factor in amplification_factors:
#         run_intervention_experiment(
#             model, tokenizer, model_type, samples,
#             neurons_to_intervene=asian_neurons_all,
#             experiment_name="White->Asian: Intervene on Asian neurons",
#             amplification_factor=amp_factor
#         )


# if __name__ == "__main__":
#     main()



# import torch
# import numpy as np
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from tqdm import tqdm
# import json
# import collections
# import os

# # --- GLOBAL CONFIGURATION ---
# # MODEL_NAME = 'meta-llama/Llama-3.1-8B-Instruct'
# # MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.3'
# MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
# MAX_LENGTH = 512

# # Input file with misclassified samples
# MISCLASSIFIED_FILE = 'misclassified_samples_with_activations_new.jsonl'

# # Define the target labels for the experiment
# TARGET_RACES = {
#     'White': 'White',
#     'Black/AA': 'Black or African American',
#     'Asian': 'Asian'
# }

# # --- NEURONS TO INTERVENE ON ---
# ALL_RACE_NEURONS = {
#     'Asian': {
#         'Direct': [
#             (27, 13406), (26, 6943), (26, 217),
#             (25, 5187), (25, 8828), (24, 15029),
#         ],
#         'Indirect': [
#             (27, 13406), (26, 6943), (26, 229),
#             (25, 9908), (25, 2001), (24, 15029),
#         ]
#     },
#     'Black/AA': {
#         'Direct': [
#             (27, 11088), (27, 10048), (26, 2240), (26, 16596),
#             (25, 18261), (25, 1091), (24, 10230), (24, 10739),
#         ],
#         'Indirect': [
#             (24, 10230), (24, 10739),
#         ]
#     },
#     'White': {
#         'Direct': [
#             (27, 16880), (26, 17660), (24, 4157), (24, 8669),
#         ],
#         'Indirect': [
#             (27, 8780), (27, 9988), (26, 17660),
#             (25, 3012), (24, 4157), (24, 5123),
#         ]
#     }
# }

# def model_setup(model_name):
#     """Loads model and tokenizer."""
#     print(f"\nLoading model: {model_name}...")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float32,
#         device_map="auto",
#         trust_remote_code=False
#     )
#     model.eval()
#     print(f"Model loaded. Layers: {model.config.num_hidden_layers}")
#     print(f"Intermediate size: {model.config.intermediate_size}")
    
#     model_type = model.config.model_type
#     print(f"Model type: {model_type}")
    
#     return model, tokenizer, model_type


# def load_misclassified_samples(filepath, filter_ground_truth=None, filter_predicted=None):
#     """
#     Load misclassified samples from JSONL file.
#     Optionally filter by ground_truth and/or predicted race.
#     """
#     samples = []
#     # Check if file exists to avoid crash
#     if not os.path.exists(filepath):
#         print(f"Error: File {filepath} not found.")
#         return []

#     with open(filepath, 'r', encoding='utf-8') as f:
#         for line in f:
#             sample = json.loads(line.strip())
            
#             # Apply filters if specified
#             if filter_ground_truth and sample['ground_truth'] != filter_ground_truth:
#                 continue
#             if filter_predicted and sample['predicted_race'] != filter_predicted:
#                 continue
                
#             samples.append(sample)
    
#     print(f"Loaded {len(samples)} samples", end="")
#     if filter_ground_truth or filter_predicted:
#         print(f" (filtered: ground_truth={filter_ground_truth}, predicted={filter_predicted})")
#     else:
#         print()
    
#     return samples


# def get_mlp_module(model, layer_idx, model_type):
#     """Returns the MLP down_proj module for a given layer."""
#     if model_type in ['llama', 'mistral', 'qwen2']:
#         return model.model.layers[layer_idx].mlp.down_proj
#     else:
#         raise ValueError(f"Unsupported model type: {model_type}")


# def generate_with_intervention(model, tokenizer, text, neurons_to_intervene, model_type, amplification_factor):
#     """
#     Generate model output while amplifying/flipping specific neurons.
#     """
#     prompt_options = ', '.join(TARGET_RACES.values())
#     prompt = (
#         "Please read the following sentence from a clinical note. "
#         "Based on the information contained within the sentence, determine which of the following races or ethnicities the patient belongs to. "
#         "Please respond with only one option.\n\n"
#         f"Options: [{prompt_options}]\n\n"
#         f"Clinical Note: \"{text}\"\n\n"
#         "Inferred Race or Ethnicity:"
#     )

#     hooks = []
    
#     # Create intervention hooks
#     def make_intervention_hook(layer_idx, neuron_indices, amp_factor):
#         def hook(module, input, output):
#             modified_input = input[0].clone()

#             for neuron_idx in neuron_indices:
#                 activation = modified_input[:, :, neuron_idx]

#                 # Flip and amplify logic:
#                 modified_input[:, :, neuron_idx] = torch.where(
#                     activation < 0,
#                     activation * amp_factor,      # negative -> more negative
#                     activation * -amp_factor      # positive -> flip to negative
#                 )

#             # Recompute output with modified input
#             # down_proj: (intermediate_size) -> (hidden_size)
#             new_output = torch.nn.functional.linear(modified_input, module.weight, module.bias)
#             return new_output
#         return hook
    
#     # Group neurons by layer
#     neurons_by_layer = collections.defaultdict(list)
#     for layer_idx, neuron_idx in neurons_to_intervene:
#         neurons_by_layer[layer_idx].append(neuron_idx)
    
#     # Register hooks
#     for layer_idx, neuron_indices in neurons_by_layer.items():
#         mlp_module = get_mlp_module(model, layer_idx, model_type)
#         # Pass the amplification_factor explicitly here
#         hook = mlp_module.register_forward_hook(
#             make_intervention_hook(layer_idx, neuron_indices, amplification_factor)
#         )
#         hooks.append(hook)
    
#     try:
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
        
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=10,
#                 do_sample=False,
#                 pad_token_id=tokenizer.eos_token_id
#             )
        
#         output_tokens = outputs[0, inputs.input_ids.shape[-1]:]
#         raw_output = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        
#         # Extract prediction
#         predicted_race_key = None
#         for race_key, race_label in TARGET_RACES.items():
#             if race_label.lower() in raw_output.lower() or race_key.lower() in raw_output.lower():
#                 predicted_race_key = race_key
#                 break
        
#     finally:
#         # Remove all hooks
#         for hook in hooks:
#             hook.remove()
    
#     return predicted_race_key, raw_output


# def run_intervention_experiment(model, tokenizer, model_type, samples, neurons_to_intervene, experiment_name, amplification_factor):
#     """
#     Run intervention experiment on a set of samples with a specific amplification factor.
#     """
#     print(f"\n{'='*80}")
#     print(f"   INTERVENTION EXPERIMENT: {experiment_name}")
#     print(f"   Amplification Factor: {amplification_factor}")
#     print(f"   Intervening on {len(neurons_to_intervene)} neurons")
#     print(f"{'='*80}")
    
#     # Track changes: {new_prediction: count}
#     change_counts = collections.defaultdict(int)

#     # Track sample outputs for Unknown and White (max 5 each)
#     unknown_outputs = []
#     white_outputs = []
#     max_samples = 3

#     for sample in tqdm(samples, desc=f"Running Factor {amplification_factor}"):
#         text = sample['sentence_text']

#         # Run with intervention
#         new_prediction, new_output = generate_with_intervention(
#             model, tokenizer, text, neurons_to_intervene, model_type, amplification_factor
#         )

#         # Track results
#         if new_prediction is None:
#             change_counts['Unknown'] += 1
#             if len(unknown_outputs) < max_samples:
#                 unknown_outputs.append({
#                     'raw_output': new_output,
#                     'sentence_text': text
#                 })
#         else:
#             change_counts[new_prediction] += 1
#             # Track White predictions
#             if new_prediction == 'White' and len(white_outputs) < max_samples:
#                 white_outputs.append({
#                     'raw_output': new_output,
#                     'sentence_text': text
#                 })

#         # Cleanup
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
    
#     # Print results table
#     total = len(samples)
#     print(f"\n{'='*60}")
#     print(f"   RESULTS: {experiment_name} | Factor: {amplification_factor}")
#     print(f"{'='*60}")
#     print(f"   Original Prediction: '{samples[0]['predicted_race']}'")
#     print(f"   Ground Truth: '{samples[0]['ground_truth']}'")
#     print(f"   Total samples: {total}")
#     print(f"{'='*60}")
#     print(f"   {'New Prediction':<20} | {'Count':<10} | {'Percentage':<10}")
#     print(f"   {'-'*50}")
    
#     for prediction in ['White', 'Black/AA', 'Asian', 'Unknown']:
#         count = change_counts.get(prediction, 0)
#         pct = 100 * count / total if total > 0 else 0
        
#         marker = ""
#         if prediction == samples[0]['ground_truth']:
#             marker = " <-- Ground Truth"
#         elif prediction == samples[0]['predicted_race']:
#             marker = " <-- No Change"
            
#         print(f"   {prediction:<20} | {count:<10} | {pct:>6.1f}%{marker}")
    
#     print(f"{'='*60}")

#     # Print White outputs if any
#     if white_outputs:
#         print(f"\n{'='*60}")
#         print(f"   SAMPLE OUTPUTS FOR 'White' (Factor {amplification_factor})")
#         print(f"{'='*60}")
#         for i, sample in enumerate(white_outputs):
#             print(f"   {i+1}. Raw output: '{sample['raw_output']}'")
#             print(f"      Sentence: {sample['sentence_text'][:100]}...")
#             print()
    
#     # --- ADDED: Print Unknown outputs if any ---
#     if unknown_outputs:
#         print(f"\n{'='*60}")
#         print(f"   SAMPLE OUTPUTS FOR 'Unknown' (Factor {amplification_factor})")
#         print(f"{'='*60}")
#         for i, sample in enumerate(unknown_outputs):
#             print(f"   {i+1}. Raw output: '{sample['raw_output']}'")
#             print(f"      Sentence: {sample['sentence_text'][:100]}...")
#             print()
    
#     print(f"{'='*60}\n")

#     return change_counts


# def main():
#     print("="*80)
#     print("   NEURON INTERVENTION EXPERIMENT")
#     print("="*80)
    
#     model, tokenizer, model_type = model_setup(MODEL_NAME)

#     samples = load_misclassified_samples(
#         MISCLASSIFIED_FILE,
#         filter_ground_truth='White',
#         filter_predicted='Asian'
#     )
    
#     if not samples:
#         print("No samples found matching the filter criteria.")
#         return

#     # Define neurons to intervene on
#     black_neurons_direct = ALL_RACE_NEURONS['Asian']['Direct']
#     black_neurons_all = list(set(black_neurons_direct)) 
    
#     print(f"\nNeurons to intervene on:")
#     print(f"  Combined (unique): {black_neurons_all}")
    
#     # --- LOOP THROUGH AMPLIFICATION FACTORS ---
#     factors = [5, 10, 20]
    
#     for factor in factors:
#         run_intervention_experiment(
#             model, tokenizer, model_type, samples,
#             neurons_to_intervene=black_neurons_all,
#             experiment_name="White->Black/AA: Amplify Black/AA neurons",
#             amplification_factor=factor
#         )

# if __name__ == "__main__":
#     main()