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
import pandas as pd
import collections
from huggingface_hub import login


MODEL_NAME = 'qwen/Qwen2.5-7B-Instruct'
TEST_SPLIT = 0.1  
PROBE_LAYER_INDEX = -1 
MAX_LENGTH = 128
TOP_K_TOKENS_FROM_PROBE = 20
TOP_K_NEURONS_TO_ANALYZE = 20
TOP_K_TOKENS_PER_NEURON = 10

OUTPUT_DIR = "race_probe_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


INDICATORS_PATH = 'indicators_df.jsonl'
ASSIGNMENTS_PATH = 'all_re_assignments_df.jsonl'

RACE_GROUPS = {
    'white': ['RACE White'],
    'black_aa': ['RACE Black or African American'],
    'asian': ['RACE Asian'],
    'native_am_ak': ['RACE American Indian or Alaska Native'],
    'native_hi_pi': ['RACE Native Hawaiian or Other Pacific Islander']
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

def load_creact_direct_probe_data(race_groups):
    print("\n--- Starting C-REACT Data Preprocessing and Filtering ---")


    df_indicators = pd.read_json(INDICATORS_PATH, lines=True)
    df_assignments = pd.read_json(ASSIGNMENTS_PATH, lines=True)
    
    df_merged = df_indicators.merge(df_assignments, on='sentence_id', suffixes=('_ind', '_assign'))
    print(f"Merged DataFrames: {len(df_merged)} sentences.")

    all_patients = set(df_merged['patient_id'].unique())
    
    # Identify patients with  direct or indirect indicators
    direct_labels = ['race', 'ethnicity']
    indirect_labels = ['country', 'language']
    
    def has_indicator_type(spans, types):
        return any(span['label'] in types for span in spans)

    patients_with_indirect_mentions = set(
        df_merged[df_merged['spans'].apply(
            lambda s: has_indicator_type(s, indirect_labels)
        )]['patient_id'].unique()
    )
    
    patients_with_direct_mentions = set(
        df_merged[df_merged['spans'].apply(
            lambda s: has_indicator_type(s, direct_labels)
        )]['patient_id'].unique()
    )

    # Get direct mentions ONLY patients
    direct_only_patients = patients_with_direct_mentions - patients_with_indirect_mentions

    #Get indirect mentions ONLY patients (for comparison)
    # indirect_only_patients = patients_with_indirect_mentions - patients_with_direct_mentions
    # direct_only_patients = indirect_only_patients
    
    print(f"Total Unique Patients: {len(all_patients)}")
    print(f"Patients with ONLY Direct Mentions (Filter Group): {len(direct_only_patients)}")

    df_filtered_patients = df_merged[
        df_merged['patient_id'].isin(direct_only_patients)
    ].copy()
    
    print(f"Sentences remaining after patient filter: {len(df_filtered_patients)}")
    
    race_cols = [name for name in df_filtered_patients.columns if name.startswith('RACE ')]
    
    df_labeled = df_filtered_patients[df_filtered_patients['RACE No Information Indicated'] == 0].copy()
    
    race_cols_positive = [c for c in race_cols if c not in ['RACE No Information Indicated', 'RACE Not Covered']]

    print("Detected Positive RACE Columns:", race_cols_positive)
    
    # Identify the single positive RACE label for each sentence
    def get_single_race_label(row):
        for col in race_cols_positive:
            if row[col] == 1:
                return col
        return None
    
    df_labeled['race_label_col'] = df_labeled.apply(get_single_race_label, axis=1)
    
    target_race_names = [group[0] for group in race_groups.values()]
    df_final = df_labeled[df_labeled['race_label_col'].isin(target_race_names)].copy()
    
    group_to_label = {name[0]: i for i, name in enumerate(race_groups.values())}
    label_to_group = {i: name for i, name in enumerate(race_groups.keys())}
    
    df_final['label'] = df_final['race_label_col'].apply(lambda x: group_to_label.get(x))

    filtered_data = []
    group_counts = collections.Counter()
    
    for index, row in df_final.iterrows():
        filtered_data.append({
            'text': row['text_ind'],
            'label': row['label'],
            'group_name': label_to_group[row['label']],
            'patient_id': row['patient_id'] 
        })
        group_counts[label_to_group[row['label']]] += 1
        
    print(f"\nTotal samples for Probe (Sentences): {len(filtered_data)}")
    print("Distribution by final Race Group:")
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

    unique_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))

    target_names = [label_to_group[i] for i in unique_labels]
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, 
        y_pred, 
        labels=unique_labels, 
        target_names=target_names 
    ))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    return probe, accuracy

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
    
    probe_path = os.path.join(OUTPUT_DIR, f"test.pkl")
    with open(probe_path, 'wb') as f:
        pickle.dump(probe_data, f)
    
    print(f"\n Probe saved to: {probe_path}")
    
    return probe_path

## Print out the top 20 tokens projected by each race probe
def validate_probe_with_logitlens(probe, tokenizer, model, label_to_group, unique_labels):
    logit_probe = probe.named_steps['logisticregression']

    unembedding_matrix = model.get_output_embeddings().weight.data
    final_norm = model.model.norm 
    
    all_probe_tokens = {}
    
    for class_idx in unique_labels:
        group_name = label_to_group[class_idx]
        W_race = logit_probe.coef_[class_idx]
        
        W_race_tensor = torch.tensor(W_race, dtype=unembedding_matrix.dtype, 
                                      device=unembedding_matrix.device)
        
        # apply layer norm to probe then project
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
def analyze_race_neurons_by_cosine_similarity(probe, model, tokenizer, probe_layer_idx, label_to_group, unique_labels):
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
    
    for class_idx in unique_labels:
        group_name = label_to_group[class_idx]
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
    
    dataset, label_to_group = load_creact_direct_probe_data(RACE_GROUPS)
    
    X, y = create_multiclass_probe_dataset(model, tokenizer, dataset, PROBE_LAYER_INDEX)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
    )
   
    probe, accuracy = train_multiclass_probe(X_train, y_train, X_test, y_test, label_to_group)

    unique_labels = sorted(np.unique(np.concatenate([y_train, y_test])))
    

    probe_path = save_probe(probe, MODEL_NAME, accuracy, label_to_group)

    probe_tokens = validate_probe_with_logitlens(probe, tokenizer, model, label_to_group, unique_labels)

    neuron_results = analyze_race_neurons_by_cosine_similarity(
        probe, model, tokenizer, PROBE_LAYER_INDEX, label_to_group, unique_labels
    )
    
    print(f"\nProbe saved to: {probe_path}")


if __name__ == "__main__":
    main()
