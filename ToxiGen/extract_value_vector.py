import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np

from modeling_qwen2_mlp import Qwen2Config, Qwen2ForCausalLM

AutoModelForCausalLM.register(Qwen2Config, Qwen2ForCausalLM, exist_ok=True)


MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.7
# This is the 'α' (alpha) value that we can tune
INTERVENTION_STRENGTH = 1.0 


RACE_NEURONS = {
    'asian': [
        (27, 13406), 
        (27, 5983),  
        (26, 8641),  
        (26, 668),   
        (24, 15029), 
    ],
    'latino': [
        (27, 4781),  
        (27, 9876),  
        (26, 18125), 
        (24, 1321),  
    ],
    'middle_eastern': [
        (27, 9988),  
        (26, 9840),  
        (26, 8005),  
        (25, 13473), 
        (25, 3012),  
        (24, 2602),  
    ]
}


TOXIGEN_TO_RACE = {
    'asian': 'asian',
    'chinese': 'asian',
    'latino': 'latino',
    'mexican': 'latino',
    'middle_east': 'middle_eastern',
    'jewish': 'middle_eastern',
    'muslim': 'middle_eastern',
}


def model_setup(model_name):
    print(f"Loading model: {model_name}...")
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

## select one harmful prompt for each race group from ToxiGen annotations
def select_prompts_from_toxigen():
    # This uses the 'annotated' subset as requested
    dataset = load_dataset("toxigen/toxigen-data", "annotations", split="train")
    
    selected_prompts = {}
    
    for entry in dataset:
        # Input.prompt_label = 1 is "harm"
        if entry['Input.prompt_label'] != 1:
            continue
        
        target_group = entry['Inut.target_group']
        text = entry['Input.text']
        
        if target_group not in TOXIGEN_TO_RACE:
            continue
        
        race_group = TOXIGEN_TO_RACE[target_group]
        
        if race_group not in selected_prompts:
            selected_prompts[race_group] = {
                'text': text,
                'target_group': target_group
            }
        
        if all(g in selected_prompts for g in RACE_NEURONS.keys()):
            break
            
    print("\n--- Selected Prompts ---")
    for race, info in selected_prompts.items():
        print(f"\n{race.upper()} (from {info['target_group']}):")
        print(f"  \"{info['text'][:100]}...\"")
        
    return selected_prompts

##  It subtracts a specific layer's value vector (v_i) from the final layer's hidden state (x^{L-1}), right before the lm_head calculates the logits
class FinalLayerIntervener:
    def __init__(self, model, layer_idx, neuron_idx, alpha=1.0):
        self.model = model
        self.layer_idx = layer_idx
        self.neuron_idx = neuron_idx
        self.alpha = alpha
        self.hook_handle = None
        self.v_i = None

        if self.alpha != 0.0:
            try:
                mlp = self.model.model.layers[self.layer_idx].mlp
                self.v_i = mlp.down_proj.weight[:, self.neuron_idx].detach()
            except Exception as e:
                print(f"Error accessing neuron {neuron_idx} in layer {layer_idx}: {e}")
                raise

    def intervention_hook(self, module, input_args):
        hidden_states = input_args[0]
        
        if self.v_i is not None and self.alpha != 0.0:
            
            # x^{L-1} * v_i
            intervention_vector = self.alpha * self.v_i
            
            # Subtract from the last token's hidden state
            hidden_states[:, -1, :] = hidden_states[:, -1, :] - intervention_vector

        return (hidden_states,)

    def __enter__(self):
        if self.alpha != 0.0:
            self.hook_handle = self.model.lm_head.register_forward_pre_hook(self.intervention_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_handle is not None:
            self.hook_handle.remove()

## Get a baseline output
def generate_with_intervention(model, tokenizer, prompt, layer_idx, neuron_idx, alpha=0.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with FinalLayerIntervener(model, layer_idx, neuron_idx, alpha=alpha):
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    continuation = generated_text[len(prompt):].strip()
    
    return continuation


def run_neuron_manipulation_experiment(model, tokenizer, prompts):
    print("\n" + "="*80)
    print("      RUNNING NEURON MANIPULATION EXPERIMENT")
    print("="*80)
    
    for race_group, prompt_info in prompts.items():
        if race_group not in RACE_NEURONS:
            print(f"\nSkipping {race_group}, no neurons defined.")
            continue
            
        prompt = prompt_info['text']
        neurons = RACE_NEURONS[race_group]
        
        print(f"\n{'#'*80}")
        print(f"    RACE GROUP: {race_group.upper()}")
        print(f"{'#'*80}")
        print(f"\nPROMPT: \"{prompt}\"")
        
        print(f"\n{'='*80}")
        print("  BASELINE (No Intervention, alpha=0.0)")
        print(f"{'='*80}")
        
        baseline_output = generate_with_intervention(
            model, tokenizer, prompt, 
            layer_idx=0, neuron_idx=0, alpha=0.0
        )
        print(f"{baseline_output}")
        
        # Test each neuron for each group 
        print(f"\n{'='*80}")
        print(f"  INTERVENTIONS (alpha={INTERVENTION_STRENGTH})")
        print(f"{'='*80}")
        
        for idx, (layer_idx, neuron_idx) in enumerate(neurons, 1):
            print(f"\n--- Manipulating Neuron {idx}/{len(neurons)}: (Layer {layer_idx}, Neuron {neuron_idx}) ---")
            
            try:
                suppressed_output = generate_with_intervention(
                    model, tokenizer, prompt,
                    layer_idx=layer_idx,
                    neuron_idx=neuron_idx,
                    alpha=INTERVENTION_STRENGTH 
                )
                print(f"→ {suppressed_output}")
            
            except Exception as e:
                print(f"Error during generation with intervention: {e}")

    print("\n" + "="*80)
    print("      EXPERIMENT COMPLETE")
    print("="*80)


def main():
    model, tokenizer = model_setup(MODEL_NAME)
    
    prompts = select_prompts_from_toxigen()
    
    required_groups = RACE_NEURONS.keys()
    missing_groups = [g for g in required_groups if g not in prompts]
    
    if missing_groups:
        print(f"\n Warning: Could not find harmful prompts for all groups: {missing_groups}")
    
    run_neuron_manipulation_experiment(model, tokenizer, prompts)


if __name__ == "__main__":
    main()