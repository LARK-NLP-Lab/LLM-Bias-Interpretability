import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
TEST_SPLIT = 0.1  
PROBE_LAYER_INDEX = -1 
MAX_LENGTH = 128
TOP_K_TOKENS_FROM_PROBE = 20
TOP_K_NEURONS_TO_ANALYZE = 20
TOP_K_TOKENS_PER_NEURON = 10

OUTPUT_DIR = "race_probe_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RACE_GROUPS = {
    'asian': ['asian', 'chinese'],
    'black': ['black'],
    'latino': ['latino', 'mexican'],
    'native_american': ['native_american'],
    'middle_eastern': ['middle_east', 'jewish', 'muslim']
}


def model_setup(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model.eval()
    print("Model loaded.")
    print(f"Number of layers: {model.config.num_hidden_layers}")
    return model, tokenizer

## Load dataset and map numeric labels to race groups
def load_toxigen_multiclass(race_groups):
    dataset = load_dataset("toxigen/toxigen-data", name="annotated", split="train")
    
    label_to_group = {i: name for i, name in enumerate(race_groups.keys())}
    group_to_label = {name: i for i, name in label_to_group.items()}
    
    all_target_groups = []
    for group_name, targets in race_groups.items():
        all_target_groups.extend(targets)
    
    filtered_data = []
    group_counts = {name: 0 for name in race_groups.keys()}
    
    for entry in dataset:

        target_group = entry['target_group'] 
        text = entry['text']
        
        if target_group not in all_target_groups:
            continue
        
        if not text or len(text.strip()) == 0:
            continue
        
        group_name = None
        for gname, targets in race_groups.items():
            if target_group in targets:
                group_name = gname
                break
        
        if group_name is None:
            continue
        
        label = group_to_label[group_name]
        group_counts[group_name] += 1
        
        filtered_data.append({
            'text': text,
            'label': label,
            'group_name': group_name,
            'target_group': target_group 
        })
    
    print(f"\nTotal samples: {len(filtered_data)}")
    print("Distribution by race group:")
    for group_name, count in group_counts.items():
        print(f"  {group_name}: {count}")
    
    return filtered_data, label_to_group

def create_multiclass_probe_dataset(model, tokenizer, dataset, probe_layer_idx):
    print(f"\nCreating probe dataset from layer {probe_layer_idx + 1} residual stream...")

    X_activations = []
    y_labels = []
    failed_samples = 0

    for idx, entry in enumerate(tqdm(dataset, desc="Generating Activations")):
        text = entry['text']
        label = entry['label']

        if not text or len(text.strip()) == 0:
            failed_samples += 1
            continue

        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # extract activations from specific layer residual stream
            if probe_layer_idx == -1:
                hidden_states = outputs.hidden_states[-1]
            else:
                hidden_states = outputs.hidden_states[probe_layer_idx]

            # Average over sequence length
            hidden_states = hidden_states.to(torch.float32)
            avg_activation = hidden_states.mean(dim=1)[0].cpu().numpy()

            if avg_activation.ndim != 1:
                failed_samples += 1
                continue

            X_activations.append(avg_activation)
            y_labels.append(label)

        except Exception as e:
            print(f"\nERROR at sample {idx}: {e}")
            failed_samples += 1
            continue
        finally:
            if 'inputs' in locals():
                del inputs
            if 'outputs' in locals():
                del outputs
            if 'hidden_states' in locals():
                del hidden_states
            if torch.cuda.is_available() and idx % 100 == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    print(f"\nTotal samples processed: {len(X_activations)}")
    print(f"Failed samples: {failed_samples}")

    X_array = np.array(X_activations)
    y_array = np.array(y_labels)

    print(f"X shape: {X_array.shape}")
    print(f"y shape: {y_array.shape}")

    return X_array, y_array

## Train multi-class linear probe useing logistic regression and evaluate
def train_multiclass_probe(X_train, y_train, X_test, y_test, label_to_group):
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight='balanced', max_iter=500, random_state=42, multi_class='multinomial')
    )
    
    print("Training probe...")
    probe.fit(X_train, y_train)

    y_pred = probe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    
    print(f"\n--- Probe Performance ---")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    target_names = [label_to_group[i] for i in sorted(label_to_group.keys())]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    return probe, accuracy

## Saves the trained probe for future analyses
def save_probe(probe, model_name, accuracy, label_to_group):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    probe_data = {
        'probe': probe,
        'model_name': model_name,
        'probe_layer': PROBE_LAYER_INDEX,
        'test_accuracy': accuracy,
        'label_to_group': label_to_group,
        'race_groups': RACE_GROUPS,
        'timestamp': timestamp,
        'max_length': MAX_LENGTH,
        'test_split': TEST_SPLIT
    }
    
    probe_path = os.path.join(OUTPUT_DIR, f"race_probe_{timestamp}.pkl")
    with open(probe_path, 'wb') as f:
        pickle.dump(probe_data, f)
    
    print(f"\n Probe saved to: {probe_path}")
    
    return probe_path

## Print out the top 20 tokens projected by each race probe
def validate_probe_with_logitlens(probe, tokenizer, model, label_to_group):
    logit_probe = probe.named_steps['logisticregression']

    unembedding_matrix = model.get_output_embeddings().weight.data
    final_norm = model.model.norm 
    
    all_probe_tokens = {}
    
    for class_idx, group_name in label_to_group.items():
        W_race = logit_probe.coef_[class_idx]
        
        W_race_tensor = torch.tensor(W_race, dtype=unembedding_matrix.dtype, 
                                      device=unembedding_matrix.device)
        
        # apply layer norm to probe then project (the same as what the toxicity paper does)
        W_race_normed = final_norm(W_race_tensor)
        logit_lens_scores = unembedding_matrix @ W_race_normed
        
        # get top tokens
        top_tokens = torch.topk(logit_lens_scores, TOP_K_TOKENS_FROM_PROBE)
        decoded_tokens = [tokenizer.decode([tok_id]) for tok_id in top_tokens.indices]
        
        print(f"\n--- Top {TOP_K_TOKENS_FROM_PROBE} tokens for '{group_name}' ---")
        for i, (token, score) in enumerate(zip(decoded_tokens, top_tokens.values), 1):
            print(f"  {i:2d}. '{token}' (score: {score.item():.4f})")
        
        all_probe_tokens[group_name] = decoded_tokens
    
    print("="*60)
    
    return all_probe_tokens

## Find race-related neurons using cosine similarity
def analyze_race_neurons_by_cosine_similarity(probe, model, tokenizer, probe_layer_idx, label_to_group):
    print("\n" + "="*60)
    print("Race Neuron Analysis by Cosine Similarity")
    print("="*60)

    logit_probe = probe.named_steps['logisticregression']
    
    # get value vectors
    V_matrix = model.model.layers[probe_layer_idx].mlp.down_proj.weight.data
    unembedding_matrix = model.get_output_embeddings().weight.data
    final_norm = model.model.norm
    
    print(f"Total neurons: {V_matrix.shape[1]:,}")
    
    all_neuron_results = {}
    
    for class_idx, group_name in label_to_group.items():
        W_race = logit_probe.coef_[class_idx]
        
        W_race_tensor = torch.tensor(W_race, dtype=V_matrix.dtype, device=V_matrix.device)
        W_race_normalized = W_race_tensor / torch.norm(W_race_tensor)
        
        # calculate cosine similarities for all neurons
        similarities = []
        for i in range(V_matrix.shape[1]):
            v_i = V_matrix[:, i]
            v_i_normalized = v_i / torch.norm(v_i)
            cos_sim = torch.dot(W_race_normalized, v_i_normalized).item()
            similarities.append(cos_sim)
        
        similarities = np.array(similarities)
        top_256_indices = np.argsort(similarities)[-256:][::-1]
        
        print(f"\n--- Top {TOP_K_NEURONS_TO_ANALYZE} '{group_name}' Neurons ---")
        
        neuron_results = []
        for rank, neuron_idx in enumerate(top_256_indices[:TOP_K_NEURONS_TO_ANALYZE]):
            v_i = V_matrix[:, neuron_idx]
            
            v_i_normed = final_norm(v_i)
            logit_lens_scores = unembedding_matrix @ v_i_normed
            
            top_tokens = torch.topk(logit_lens_scores, TOP_K_TOKENS_PER_NEURON)
            decoded_tokens = [tokenizer.decode([tok_id]) for tok_id in top_tokens.indices]
            
            print(f"  Rank {rank+1:2d} | Neuron {neuron_idx:5d} | Cosine Similarity: {similarities[neuron_idx]:.6f}")
            print(f"         Top Tokens: {decoded_tokens}")
            
            neuron_results.append({
                'rank': rank + 1,
                'neuron_idx': neuron_idx,
                'cosine_similarity': similarities[neuron_idx],
                'top_tokens': decoded_tokens
            })    
        
        all_neuron_results[group_name] = {
            'top_neurons': neuron_results,
            'top_256_indices': top_256_indices,
            'similarities': similarities
        }
    
    print("="*60)
    return all_neuron_results


def main():
    print("="*60)
    print("   Multi-class Race Group Linear Probe Analysis")
    print("="*60)
    
    model, tokenizer = model_setup(MODEL_NAME)
    
    dataset, label_to_group = load_toxigen_multiclass(RACE_GROUPS)
    
    X, y = create_multiclass_probe_dataset(model, tokenizer, dataset, PROBE_LAYER_INDEX)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
    )
   
    probe, accuracy = train_multiclass_probe(X_train, y_train, X_test, y_test, label_to_group)
    

    probe_path = save_probe(probe, MODEL_NAME, accuracy, label_to_group)

    probe_tokens = validate_probe_with_logitlens(probe, tokenizer, model, label_to_group)

    neuron_results = analyze_race_neurons_by_cosine_similarity(
        probe, model, tokenizer, PROBE_LAYER_INDEX, label_to_group
    )
    
    print(f"\nProbe saved to: {probe_path}")


if __name__ == "__main__":
    main()