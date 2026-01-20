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
    
    login('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')
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

# # calculate activations
# import torch
# import numpy as np
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from tqdm import tqdm
# import pandas as pd
# import collections
# from modeling_qwen2_mlp import Qwen2Config, Qwen2ForCausalLM 

# AutoModelForCausalLM.register(Qwen2Config, Qwen2ForCausalLM, exist_ok=True)

# MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
# MAX_LENGTH = 512 
# N_SAMPLES_PER_GROUP = 100 

# PROMPT_TEMPLATE = (
#     "Read the following clinical note of a patient:\n"
#     "\"{text}\"\n\n"
#     "Based on the description above, what is the most likely race or ethnicity of the patient? Please choose between Asian, Black or African American and White\n"
#     "Answer with only your choice of race:"
# )

# INDICATORS_PATH = 'indicators_df.jsonl'
# ASSIGNMENTS_PATH = 'all_re_assignments_df.jsonl'

# C_REACT_RACE_GROUPS = {
#     'White': 'RACE White',
#     'Black/AA': 'RACE Black or African American',
#     'Asian': 'RACE Asian',
# }

# RACE_NEURONS = {
#     'asian': {
#         'direct': [
#             (27, 13406), (27, 16570), (26, 6943), (26, 217),
#             (25, 5187), (25, 8828), (24, 15029),
#         ],
#         'indirect': [
#             (27, 13406), (26, 6943), (26, 229),
#             (25, 9908), (25, 2001), (24, 15029),
#         ]
#     },
#     'black': {
#         'direct': [
#             (27, 11088), (27, 10048), (26, 2240), (26, 16596),
#             (25, 18261), (25, 1091), (24, 10230), (24, 10739),
#         ],
#         'indirect': [
#             (24, 10230), (24, 10739),
#         ]
#     },
#     'white': {
#         'direct': [
#             (27, 16880), (26, 17660), (24, 4157), (24, 8669),
#         ],
#         'indirect': [
#             (27, 8780), (27, 9988), (27, 4318), (26, 17660),
#             (25, 3012), (24, 4157), (24, 5123),
#         ]
#     }
# }

# def model_setup(model_name):
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
#     return model, tokenizer

# def load_creact_by_mention_type(n_samples_per_group):
#     print("\n--- Starting C-REACT Data Loading ---")
#     try:
#         df_indicators = pd.read_json(INDICATORS_PATH, lines=True)
#         df_assignments = pd.read_json(ASSIGNMENTS_PATH, lines=True)
#     except FileNotFoundError as e:
#         print(f"ERROR: Data file not found. Check paths: {e}")
#         return {}

#     df_merged = df_indicators.merge(df_assignments, on='sentence_id', suffixes=('_ind', '_assign'))
    
#     direct_labels = ['race', 'ethnicity']
#     indirect_labels = ['country', 'language']
    
#     def has_indicator_type(spans, types):
#         return any(span.get('label') in types for span in spans)

#     df_merged['is_direct'] = df_merged['spans'].apply(lambda s: has_indicator_type(s, direct_labels))
#     df_merged['is_indirect'] = df_merged['spans'].apply(lambda s: has_indicator_type(s, indirect_labels))
    
#     df_merged['mention_type'] = 'Zero Mentions'
#     df_merged.loc[df_merged['is_direct'] & ~df_merged['is_indirect'], 'mention_type'] = 'Direct Only'
#     df_merged.loc[~df_merged['is_direct'] & df_merged['is_indirect'], 'mention_type'] = 'Indirect Only'
    
#     def get_assigned_race(row):
#         for simple_name, col_name in C_REACT_RACE_GROUPS.items():
#             if row.get(col_name) == 1:
#                 return simple_name
#         return None 

#     df_merged['assigned_race'] = df_merged.apply(get_assigned_race, axis=1)
#     df_merged.loc[df_merged['mention_type'] == 'Zero Mentions', 'assigned_race'] = None

#     final_texts = collections.defaultdict(list)
#     df_final_groups = df_merged[df_merged['assigned_race'].notna()].copy()

#     for (race_name, mention_type), group_df in df_final_groups.groupby(['assigned_race', 'mention_type']):
#         group_key = f"{race_name}_{mention_type}" 
#         texts = group_df['text_ind'].tolist()

#         if len(texts) > n_samples_per_group:
#             np.random.seed(42)
#             indices = np.random.choice(len(texts), n_samples_per_group, replace=False)
#             texts = [texts[i] for i in indices]

#         final_texts[group_key] = texts
    
#     zero_mention_texts = df_merged[df_merged['mention_type'] == 'Zero Mentions']['text_ind'].tolist()
#     if len(zero_mention_texts) > n_samples_per_group:
#         np.random.seed(42)
#         indices = np.random.choice(len(zero_mention_texts), n_samples_per_group, replace=False)
#         zero_mention_texts = [zero_mention_texts[i] for i in indices]

#     final_texts['Zero Mentions_No Race Label'] = zero_mention_texts
    
#     print(f"Loaded {len(final_texts)} text groups.")
#     return final_texts

# def extract_neuron_activations(model, tokenizer, texts, neurons_to_track):
#     neurons_by_layer = {}
#     for layer_idx, neuron_idx in neurons_to_track:
#         if layer_idx not in neurons_by_layer:
#             neurons_by_layer[layer_idx] = []
#         neurons_by_layer[layer_idx].append(neuron_idx)
    
#     neuron_activations = {neuron: [] for neuron in neurons_to_track}
    
#     for text in tqdm(texts, desc="Extracting activations", leave=False):
#         full_prompt = PROMPT_TEMPLATE.format(text=text)
#         inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
        
#         with torch.no_grad():
#             outputs = model(**inputs, output_mlp_pre_residual=True) 
        
#         all_mlp_activations = outputs.mlp_outputs
        
#         for layer_idx, neuron_indices in neurons_by_layer.items():
#             layer_activations = all_mlp_activations[layer_idx]
#             for neuron_idx in neuron_indices:
#                 # avg_act = layer_activations[0, :, neuron_idx].mean().item()
#                 # neuron_activations[(layer_idx, neuron_idx)].append(avg_act)
#                 last_token_act = layer_activations[0, -1, neuron_idx].item()
#                 neuron_activations[(layer_idx, neuron_idx)].append(last_token_act)
        
#         del inputs, outputs
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
    
#     return neuron_activations

# def compare_cross_race_activations(model, tokenizer, comparison_texts, race_neurons):
#     print("\n" + "="*80)
#     print("    FULL C-REACT ACTIVATION ANALYSIS")
#     print("    1. Intra-Race: Direct vs Indirect Neurons")
#     print("    2. Inter-Race: Specificity Check (Asian vs Black vs White Neurons)")
#     print("="*80)
    
#     # Map for display names vs keys
#     race_key_map = {
#         'asian': 'Asian',
#         'black': 'Black/AA',
#         'white': 'White'
#     }

#     results_summary = {}

#     # Iterate through each Race Context (e.g., Asian Text, Black Text...)
#     for context_race_key, context_display_name in race_key_map.items():
#         if context_race_key not in race_neurons: 
#             continue

#         print(f"\n\n{'='*60}")
#         print(f"   CONTEXT: {context_display_name.upper()} INDIRECT TEXTS")
#         print(f"{'='*60}")
        
#         # 1. Get the INDIRECT Texts for this context
#         target_text_key = f"{context_display_name}_Indirect Only"
#         target_texts = comparison_texts.get(target_text_key, [])
        
#         if not target_texts:
#             print(f"  [WARNING] No Indirect texts found for {context_display_name}")
#             continue
            
#         print(f"  Analyzing {len(target_texts)} samples...")

#         # --- ANALYSIS PART 1: INTRA-RACE (Direct vs Indirect Mechanism) ---
#         print(f"\n  [Test 1: Mechanism Check] (Does {context_display_name} Indirect Mechanism activate?)")
        
#         own_direct_neurons = race_neurons[context_race_key]['direct']
#         own_indirect_neurons = race_neurons[context_race_key]['indirect']
        
#         direct_acts = extract_neuron_activations(model, tokenizer, target_texts, own_direct_neurons)
#         indirect_acts = extract_neuron_activations(model, tokenizer, target_texts, own_indirect_neurons)

#         # Calculate means (flattening all neurons)
#         avg_direct = np.mean([val for vals in direct_acts.values() for val in vals]) if direct_acts else 0
#         avg_indirect = np.mean([val for vals in indirect_acts.values() for val in vals]) if indirect_acts else 0
        
#         print(f"    {context_display_name} Direct Neurons:   {avg_direct:+.4f}")
#         print(f"    {context_display_name} Indirect Neurons: {avg_indirect:+.4f}")
#         print(f"    Diff (Ind - Dir): {avg_indirect - avg_direct:+.4f}")

#         # --- ANALYSIS PART 2: INTER-RACE (Specificity Check) ---
#         print(f"\n  [Test 2: Specificity Check] (Ranking all Neuron Groups on {context_display_name} Indirect Text)")
        
#         # We will collect tuples of (Label, Score, Is_Target_Mechanism)
#         all_specificity_scores = []

#         # 1. Add the Target Race's scores (Calculated in Test 1 above)
#         # We flag 'avg_indirect' as True because that is the specific hypothesis we are testing
#         all_specificity_scores.append( (f"{context_display_name} Indirect", avg_indirect, True) ) 
#         all_specificity_scores.append( (f"{context_display_name} Direct", avg_direct, False) )

#         # 2. Calculate and add Other Races' scores
#         for other_race_key, other_display_name in race_key_map.items():
#             if other_race_key == context_race_key:
#                 continue # We already added the target race above
            
#             # A. Calculate Other Indirect
#             other_ind_neurons = race_neurons[other_race_key]['indirect']
#             other_ind_acts = extract_neuron_activations(model, tokenizer, target_texts, other_ind_neurons)
#             avg_other_ind = np.mean([val for vals in other_ind_acts.values() for val in vals]) if other_ind_acts else 0
#             all_specificity_scores.append( (f"{other_display_name} Indirect", avg_other_ind, False) )

#             # B. Calculate Other Direct (NEW ADDITION)
#             other_dir_neurons = race_neurons[other_race_key]['direct']
#             other_dir_acts = extract_neuron_activations(model, tokenizer, target_texts, other_dir_neurons)
#             avg_other_dir = np.mean([val for vals in other_dir_acts.values() for val in vals]) if other_dir_acts else 0
#             all_specificity_scores.append( (f"{other_display_name} Direct", avg_other_dir, False) )

#         # 3. Sort all results by activation (High to Low)
#         all_specificity_scores.sort(key=lambda x: x[1], reverse=True)
        
#         # 4. Print the Leaderboard
#         for label, score, is_target_mech in all_specificity_scores:
#             # Mark the specific mechanism we hope to see at the top
#             marker = " <<< TARGET MECHANISM" if is_target_mech else ""
#             print(f"    {label:25s}: {score:+.4f}{marker}")

#         # Quick Logic Check for the printout
#         top_scorer_label = all_specificity_scores[0][0]
#         if all_specificity_scores[0][2]: # If the top one is our target mechanism
#             print(f"    >> SUCCESS: Target Indirect Neurons are the most active group overall.")
#         else:
#             print(f"    >> WARNING: {top_scorer_label} are more active than the target.")

#     return results_summary

# def main():
#     model, tokenizer = model_setup(MODEL_NAME)
#     comparison_texts = load_creact_by_mention_type(N_SAMPLES_PER_GROUP)
    
#     # Run the full comparison
#     compare_cross_race_activations(model, tokenizer, comparison_texts, RACE_NEURONS)

# if __name__ == "__main__":
#     main()

# import torch
# import numpy as np
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from tqdm import tqdm
# import pandas as pd
# import collections
# from modeling_qwen2_mlp import Qwen2Config, Qwen2ForCausalLM 

# AutoModelForCausalLM.register(Qwen2Config, Qwen2ForCausalLM, exist_ok=True)

# MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
# MAX_LENGTH = 10 
# N_SAMPLES_PER_GROUP = 100 

# PROMPT_TEMPLATE = (
#     "Read the following description of a patient:\n"
#     "\"{text}\"\n\n"
#     "Based on the description above, what is the most likely race or ethnicity of the patient? Please choose between Asian, Black or African American and White\\n"
#     "Answer:"
# )

# INDICATORS_PATH = 'indicators_df.jsonl'
# ASSIGNMENTS_PATH = 'all_re_assignments_df.jsonl'

# C_REACT_RACE_GROUPS = {
#     'White': 'RACE White',
#     'Black/AA': 'RACE Black or African American',
#     'Asian': 'RACE Asian',
# }

# RACE_NEURONS = {
#     'asian': {
#         'direct': [
#             (27, 13406), (26, 16570), (26, 6943), (26, 217),
#             (25, 5187), (25, 8828), (24, 15029),
#         ],
#         'indirect': [
#             (27, 13406), (26, 6943), (26, 229),
#             (25, 9908), (25, 2001), (24, 15029),
#         ]
#     },
#     'black': {
#         'direct': [
#             (27, 11088), (27, 10048), (26, 2240), (26, 16596),
#             (25, 18261), (25, 1091), (24, 10230), (24, 10739),
#         ],
#         'indirect': [
#             (24, 10230), (24, 10739),
#         ]
#     },
#     'white': {
#         'direct': [
#             (27, 16880), (26, 17660), (24, 4157), (24, 8669),
#         ],
#         'indirect': [
#             (27, 8780), (27, 9988), (27, 4318), (26, 17660),
#             (25, 3012), (24, 4157), (24, 5123),
#         ]
#     }
# }

# def model_setup(model_name):
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
#     return model, tokenizer

# def load_creact_by_mention_type(n_samples_per_group):
#     print("\n--- Starting C-REACT Data Loading ---")
#     try:
#         df_indicators = pd.read_json(INDICATORS_PATH, lines=True)
#         df_assignments = pd.read_json(ASSIGNMENTS_PATH, lines=True)
#     except FileNotFoundError as e:
#         print(f"ERROR: Data file not found. Check paths: {e}")
#         return {}

#     df_merged = df_indicators.merge(df_assignments, on='sentence_id', suffixes=('_ind', '_assign'))
    
#     direct_labels = ['race', 'ethnicity']
#     indirect_labels = ['country', 'language']
    
#     def has_indicator_type(spans, types):
#         return any(span.get('label') in types for span in spans)

#     df_merged['is_direct'] = df_merged['spans'].apply(lambda s: has_indicator_type(s, direct_labels))
#     df_merged['is_indirect'] = df_merged['spans'].apply(lambda s: has_indicator_type(s, indirect_labels))
    
#     df_merged['mention_type'] = 'Zero Mentions'
#     df_merged.loc[df_merged['is_direct'] & ~df_merged['is_indirect'], 'mention_type'] = 'Direct Only'
#     df_merged.loc[~df_merged['is_direct'] & df_merged['is_indirect'], 'mention_type'] = 'Indirect Only'
    
#     def get_assigned_race(row):
#         for simple_name, col_name in C_REACT_RACE_GROUPS.items():
#             if row.get(col_name) == 1:
#                 return simple_name
#         return None 

#     df_merged['assigned_race'] = df_merged.apply(get_assigned_race, axis=1)
#     df_merged.loc[df_merged['mention_type'] == 'Zero Mentions', 'assigned_race'] = None

#     final_texts = collections.defaultdict(list)
#     df_final_groups = df_merged[df_merged['assigned_race'].notna()].copy()

#     for (race_name, mention_type), group_df in df_final_groups.groupby(['assigned_race', 'mention_type']):
#         group_key = f"{race_name}_{mention_type}" 
#         texts = group_df['text_ind'].tolist()

#         if len(texts) > n_samples_per_group:
#             np.random.seed(42)
#             indices = np.random.choice(len(texts), n_samples_per_group, replace=False)
#             texts = [texts[i] for i in indices]

#         final_texts[group_key] = texts
    
#     zero_mention_texts = df_merged[df_merged['mention_type'] == 'Zero Mentions']['text_ind'].tolist()
#     if len(zero_mention_texts) > n_samples_per_group:
#         np.random.seed(42)
#         indices = np.random.choice(len(zero_mention_texts), n_samples_per_group, replace=False)
#         zero_mention_texts = [zero_mention_texts[i] for i in indices]

#     final_texts['Zero Mentions_No Race Label'] = zero_mention_texts
    
#     print(f"Loaded {len(final_texts)} text groups.")
#     return final_texts

# def extract_neuron_activations(model, tokenizer, texts, neurons_to_track):
#     """
#     Extracts activations at TWO positions:
#     1. Context (Mean of all tokens BEFORE the last one) -> 'thinking'
#     2. Last Token (The final token) -> 'answering'
#     """
#     neurons_by_layer = {}
#     for layer_idx, neuron_idx in neurons_to_track:
#         if layer_idx not in neurons_by_layer:
#             neurons_by_layer[layer_idx] = []
#         neurons_by_layer[layer_idx].append(neuron_idx)
    
#     # Storage: List of tuples (context_act, last_token_act)
#     neuron_activations = {neuron: [] for neuron in neurons_to_track}
    
#     for text in tqdm(texts, desc="Extracting activations (Context vs Last)", leave=False):
#         full_prompt = PROMPT_TEMPLATE.format(text=text)
#         inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
        
#         with torch.no_grad():
#             outputs = model(**inputs, output_mlp_pre_residual=True) 
        
#         all_mlp_activations = outputs.mlp_outputs
        
#         for layer_idx, neuron_indices in neurons_by_layer.items():
#             layer_activations = all_mlp_activations[layer_idx]
#             for neuron_idx in neuron_indices:
#                 # 1. Context Activation (Average of everything up to the last token)
#                 # Slicing [0, :-1, neuron_idx]
#                 if layer_activations.shape[1] > 1:
#                     context_act = layer_activations[0, :-1, neuron_idx].mean().item()
#                 else:
#                     context_act = 0.0 # Fallback for extremely short sequences

#                 # 2. Last Token Activation
#                 last_token_act = layer_activations[0, -1, neuron_idx].item()
                
#                 neuron_activations[(layer_idx, neuron_idx)].append((context_act, last_token_act))
        
#         del inputs, outputs
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
    
#     return neuron_activations

# def compare_cross_race_activations(model, tokenizer, comparison_texts, race_neurons):
#     print("\n" + "="*80)
#     print("    FULL C-REACT ACTIVATION ANALYSIS")
#     print("    Comparing 'Context Phase' (Reading) vs 'Prediction Phase' (Answering)")
#     print("="*80)
    
#     race_key_map = {
#         'asian': 'Asian',
#         'black': 'Black/AA',
#         'white': 'White'
#     }

#     # Helper to calculate stats
#     def get_stats(act_dict):
#         all_ctx = []
#         all_last = []
#         for vals in act_dict.values():
#             for (ctx, last) in vals:
#                 all_ctx.append(ctx)
#                 all_last.append(last)
#         # Return (Mean Context, Mean Last Token)
#         return np.mean(all_ctx) if all_ctx else 0, np.mean(all_last) if all_last else 0

#     for context_race_key, context_display_name in race_key_map.items():
#         if context_race_key not in race_neurons: 
#             continue

#         print(f"\n\n{'='*60}")
#         print(f"   CONTEXT: {context_display_name.upper()} INDIRECT TEXTS")
#         print(f"{'='*60}")
        
#         target_text_key = f"{context_display_name}_Indirect Only"
#         target_texts = comparison_texts.get(target_text_key, [])
        
#         if not target_texts:
#             print(f"  [WARNING] No Indirect texts found for {context_display_name}")
#             continue
            
#         print(f"  Analyzing {len(target_texts)} samples...")

#         # --- ANALYSIS PART 1: INTRA-RACE (Direct vs Indirect Mechanism) ---
#         print(f"\n  [Test 1: Timeline Check] (Context Phase vs Prediction Phase)")
#         print(f"  {'Group':<20} | {'Context (Reading)':<18} | {'Last Token (Answer)':<18} | {'Shift'}")
#         print("-" * 75)
        
#         own_direct_neurons = race_neurons[context_race_key]['direct']
#         own_indirect_neurons = race_neurons[context_race_key]['indirect']
        
#         direct_acts = extract_neuron_activations(model, tokenizer, target_texts, own_direct_neurons)
#         indirect_acts = extract_neuron_activations(model, tokenizer, target_texts, own_indirect_neurons)

#         ctx_dir, last_dir = get_stats(direct_acts)
#         ctx_ind, last_ind = get_stats(indirect_acts)
        
#         print(f"  {context_display_name + ' Direct':<20} | {ctx_dir:+.4f}             | {last_dir:+.4f}             | {last_dir - ctx_dir:+.4f}")
#         print(f"  {context_display_name + ' Indirect':<20} | {ctx_ind:+.4f}             | {last_ind:+.4f}             | {last_ind - ctx_ind:+.4f}")

#         # --- ANALYSIS PART 2: INTER-RACE (Specificity Check on PREDICTION Phase) ---
#         # UPDATED: Now ranking based on the LAST TOKEN activation
#         print(f"\n  [Test 2: Specificity Leaderboard] (Ranking based on PREDICTION Phase / Last Token)")
        
#         all_specificity_scores = []

#         # 1. Add Target Race's scores (Calculated above in Test 1)
#         # We use 'last_ind' and 'last_dir' now
#         all_specificity_scores.append( (f"{context_display_name} Indirect", last_ind, True) ) 
#         all_specificity_scores.append( (f"{context_display_name} Direct", last_dir, False) )

#         # 2. Add Other Races' scores
#         for other_race_key, other_display_name in race_key_map.items():
#             if other_race_key == context_race_key:
#                 continue 
            
#             # Indirect
#             other_ind_neurons = race_neurons[other_race_key]['indirect']
#             other_ind_acts = extract_neuron_activations(model, tokenizer, target_texts, other_ind_neurons)
#             _, o_last_ind = get_stats(other_ind_acts) # Ignore context, take last
#             all_specificity_scores.append( (f"{other_display_name} Indirect", o_last_ind, False) )

#             # Direct
#             other_dir_neurons = race_neurons[other_race_key]['direct']
#             other_dir_acts = extract_neuron_activations(model, tokenizer, target_texts, other_dir_neurons)
#             _, o_last_dir = get_stats(other_dir_acts) # Ignore context, take last
#             all_specificity_scores.append( (f"{other_display_name} Direct", o_last_dir, False) )

#         # Sort by activation (High to Low)
#         all_specificity_scores.sort(key=lambda x: x[1], reverse=True)
        
#         # Print Leaderboard
#         for label, score, is_target_mech in all_specificity_scores:
#             marker = " <<< TARGET" if is_target_mech else ""
#             print(f"    {label:25s}: {score:+.4f}{marker}")

#         # Final Conclusion based on Prediction Phase
#         if all_specificity_scores[0][2]:
#             print(f"    >> SUCCESS: Target neurons dominate the final decision.")
#         else:
#             print(f"    >> NOTE: {all_specificity_scores[0][0]} are strongest at the final decision.")

#     return {}

# def main():
#     model, tokenizer = model_setup(MODEL_NAME)
#     comparison_texts = load_creact_by_mention_type(N_SAMPLES_PER_GROUP)
#     compare_cross_race_activations(model, tokenizer, comparison_texts, RACE_NEURONS)

# if __name__ == "__main__":
#     main()