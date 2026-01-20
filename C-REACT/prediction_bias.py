# import torch
# import numpy as np
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from tqdm import tqdm
# import json
# import os
# import collections

# # --- GLOBAL CONFIGURATION ---
# MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
# MAX_LENGTH = 128

# # Data File Paths
# INDICATORS_PATH = 'indicators_df.jsonl'
# ASSIGNMENTS_PATH = 'all_re_assignments_df.jsonl'

# # Define the target labels for the experiment
# TARGET_RACES = {
#     'White': 'White',
#     'Black/AA': 'Black or African American',
#     'Asian': 'Asian'
# }

# # Define the C-REACT column names corresponding to the TARGET_RACES
# C_REACT_RACE_GROUPS = {
#     'White': 'RACE White',
#     'Black/AA': 'RACE Black or African American',
#     'Asian': 'RACE Asian',
# }

# # Global Tracker for Cues: {'White -> Asian': {'language': {'Spanish': 5}, ...}}
# MISCLASSIFICATION_CUE_TRACKER = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))


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
#     return model, tokenizer


# def load_creact_by_mention_type():
#     """
#     Loads ALL C-REACT sentences, filters to keep only 'Indirect Only' samples for 
#     the target races, and returns text and span data.
#     """
#     print("\n--- Starting C-REACT Data Loading (All Indirect Samples) ---")
    
#     # 1. Load DataFrames
#     try:
#         df_indicators = pd.read_json(INDICATORS_PATH, lines=True)
#         df_assignments = pd.read_json(ASSIGNMENTS_PATH, lines=True)
#     except FileNotFoundError as e:
#         print(f"ERROR: Data file not found. Check paths: {e}")
#         return {}

#     # 2. Merge Data on sentence_id
#     # Note: 'spans' usually exists only in indicators, so it won't get a suffix.
#     # 'text' might be in both, so we use suffixes.
#     df_merged = df_indicators.merge(df_assignments, on='sentence_id', suffixes=('_ind', '_assign'))
    
#     # --- 3. Classify Mentions at the Sentence Level ---
#     direct_labels = ['race', 'ethnicity']
#     indirect_labels = ['country', 'language']
    
#     def has_indicator_type(spans, types):
#         return any(span.get('label') in types for span in spans)

#     # Ensure we use the correct column for spans (usually just 'spans')
#     # If merge created 'spans_ind', use that. Otherwise use 'spans'.
#     if 'spans_ind' in df_merged.columns:
#         span_col = 'spans_ind'
#     else:
#         span_col = 'spans'

#     df_merged['is_direct'] = df_merged[span_col].apply(lambda s: has_indicator_type(s, direct_labels))
#     df_merged['is_indirect'] = df_merged[span_col].apply(lambda s: has_indicator_type(s, indirect_labels))
    
#     # 4. Determine Mention Group (Pure Groups Only)
#     df_merged['mention_type'] = 'Excluded'
    
#     # Pure Indirect Only: Must have indirect, must NOT have direct
#     df_merged.loc[~df_merged['is_direct'] & df_merged['is_indirect'], 'mention_type'] = 'Indirect Only'
    
#     # 5. Determine the Assigned Race Label
#     def get_assigned_race(row):
#         for simple_name, col_name in C_REACT_RACE_GROUPS.items():
#             if row.get(col_name) == 1:
#                 return simple_name
#         return None 

#     df_merged['assigned_race'] = df_merged.apply(get_assigned_race, axis=1)

#     # --- 6. Grouping (No Sampling Limit) ---
    
#     final_data = collections.defaultdict(list)
    
#     # Filter for sentences with positive race labels AND 'Indirect Only' mention type
#     df_final_groups = df_merged[
#         df_merged['assigned_race'].notna() & 
#         (df_merged['mention_type'] == 'Indirect Only')
#     ].copy()

#     target_race_names = list(C_REACT_RACE_GROUPS.keys())
    
#     for race_name in target_race_names:
#         group_df = df_final_groups[df_final_groups['assigned_race'] == race_name]
        
#         group_key = f"{race_name}_Indirect Only"
        
#         # We store a dictionary with both text and spans for every sample
#         # Using .to_dict('records') creates a list of dicts: [{'text_ind':..., 'spans':...}, ...]
#         # Make sure to select the correct span column name identified earlier
#         samples = group_df[['text_ind', span_col]].rename(columns={span_col: 'spans'}).to_dict('records')

#         if len(samples) > 0:
#              final_data[group_key] = samples
    
#     print("\nTotal Indirect Samples loaded per race group:")
#     total_samples = 0
#     for group_name, data in final_data.items():
#         count = len(data)
#         print(f"  {group_name}: {count} samples")
#         total_samples += count
        
#     print(f"Total samples for inference: {total_samples}")
    
#     return final_data


# def run_bias_detection(model, tokenizer, comparison_data, cue_tracker):
#     """
#     Runs classification inference, builds the confusion matrix, and tracks cues for errors.
#     """
#     print("\n--- Starting Bias Detection, Inference, and Cue Tracking ---")
    
#     race_keys = list(TARGET_RACES.keys())
#     race_labels_official = list(TARGET_RACES.values())
#     matrix_size = len(race_keys)
    
#     # Row index: Ground Truth (Actual Race); Column index: Predicted Race
#     confusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    
#     race_to_index = {race: i for i, race in enumerate(race_keys)}
#     prompt_options = ', '.join(race_labels_official)
    
#     total_processed = 0
    
#     for group_key, data_list in comparison_data.items():
        
#         ground_truth_race = group_key.split('_')[0]
#         gt_index = race_to_index[ground_truth_race]
        
#         for sample in tqdm(data_list, desc=f"Processing {group_key} ({len(data_list)} samples)"):
#             text = sample['text_ind']
#             spans = sample['spans']
            
#             # --- 1. Classification Prompt ---
#             prompt = (
#     "Please read the following sentence from a clinical note. "
#     "Based on the information contained within the sentence, determine which of the following races or ethnicities the patient belongs to. "
#     "Please respond with only one option.\n\n"
#     f"Options: [{prompt_options}]\n\n"
#     f"Clinical Note: \"{text}\"\n\n"
#     "Inferred Race or Ethnicity:"
# )
            
#             inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
            
#             with torch.no_grad():
#                 outputs = model.generate(
#                     **inputs,
#                     max_new_tokens=10,
#                     do_sample=False,
#                     pad_token_id=tokenizer.eos_token_id 
#                 )
            
#             # Slice to get only the new tokens
#             output_tokens = outputs[0, inputs.input_ids.shape[-1]:]
#             predicted_answer = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            
#             # --- 2. Extract Prediction ---
#             predicted_race_key = None
#             for race_key, race_label in TARGET_RACES.items():
#                 if race_label.lower() in predicted_answer.lower() or race_key.lower() in predicted_answer.lower():
#                     predicted_race_key = race_key
#                     break
            
#             # --- 3. Update Matrix & Track Cues ---
#             if predicted_race_key in race_to_index:
#                 pred_index = race_to_index[predicted_race_key]
#                 confusion_matrix[gt_index, pred_index] += 1
                
#                 is_correct = (predicted_race_key == ground_truth_race)
                
#                 # If Misclassified, check the spans!
#                 if not is_correct:
#                     misclass_key = f"{ground_truth_race} -> {predicted_race_key}"
                    
#                     for span in spans:
#                         label = span.get('label')
                        
#                         # FIX: Extract text using start/end offsets, not span.get('text')
#                         start = span.get('start')
#                         end = span.get('end')
#                         span_text = text[start:end].strip() if start is not None and end is not None else ''
                        
#                         # We are interested in 'country' and 'language' indicators
#                         if label in ['country', 'language'] and span_text:
#                             cue_tracker[misclass_key][label][span_text] += 1
            
#             total_processed += 1
#             del inputs, outputs
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#     return confusion_matrix, race_keys, race_labels_official, cue_tracker


# def format_results_as_table(matrix, race_keys, race_labels_official):
#     """Prints the numpy confusion matrix as a readable table."""
#     print("\n\n" + "="*80)
#     print("      MODEL CLASSIFICATION RESULTS (Confusion Matrix)")
#     print("="*80)
    
#     key_to_official_label = dict(zip(race_keys, race_labels_official))
#     display_labels = [key_to_official_label[key] for key in race_keys]
    
#     header = ["Actual \\ Predicted"] + display_labels
#     row_format = "{:<20}" + "{:<15}" * len(display_labels)
    
#     print(row_format.format(*header))
#     print("-" * (20 + 15 * len(display_labels)))
    
#     total_accuracy = 0
#     total_samples = 0
    
#     for i, actual_race_key in enumerate(race_keys):
#         actual_race_display = key_to_official_label[actual_race_key]
#         row_data = [actual_race_display] + matrix[i, :].tolist()
        
#         row_total = np.sum(matrix[i, :])
#         correct_predictions = matrix[i, i]
        
#         print(row_format.format(*row_data[:len(display_labels) + 1]))

#         if row_total > 0:
#             total_accuracy += correct_predictions
#             total_samples += row_total
    
#     print("-" * (20 + 15 * len(display_labels)))
#     predicted_totals = np.sum(matrix, axis=0)
#     total_row = ["TOTAL PREDICTED"] + predicted_totals.tolist()
#     print(row_format.format(*total_row))

#     print("-" * (20 + 15 * len(display_labels)))
#     if total_samples > 0:
#         overall_accuracy = total_accuracy / total_samples
#         print(f"Overall Test Accuracy: {overall_accuracy:.4f} ({total_accuracy}/{total_samples})")
    
#     print("="*80)


# def print_cue_analysis(cue_tracker):
#     """Prints the final statistics on misclassification cues."""
#     print("\n\n" + "="*80)
#     print("      MISCLASSIFICATION CUE ANALYSIS (What triggered the error?)")
#     print("="*80)
    
#     # Sort keys for consistent printing
#     sorted_misclass_keys = sorted(cue_tracker.keys())
    
#     if not sorted_misclass_keys:
#         print("No misclassifications recorded (or no cues found in misclassified samples).")
    
#     for misclass_key in sorted_misclass_keys:
#         cue_data = cue_tracker[misclass_key]
#         print(f"\n--- Misclassification Type: {misclass_key} ---")
        
#         # Check Language
#         if 'language' in cue_data:
#             print("  [LANGUAGE Cues]")
#             sorted_cues = sorted(cue_data['language'].items(), key=lambda x: x[1], reverse=True)
#             for cue, count in sorted_cues:
#                 print(f"    - {cue}: {count}")
        
#         # Check Country
#         if 'country' in cue_data:
#             print("  [COUNTRY Cues]")
#             sorted_cues = sorted(cue_data['country'].items(), key=lambda x: x[1], reverse=True)
#             for cue, count in sorted_cues:
#                 print(f"    - {cue}: {count}")

#     print("="*80)


# def main():
#     print("="*80)
#     print(f"   C-REACT Indirect Bias & Cue Tracking (ALL SAMPLES)")
#     print("="*80)
    
#     global MISCLASSIFICATION_CUE_TRACKER
    
#     # 1. Setup Model
#     model, tokenizer = model_setup(MODEL_NAME)

#     # 2. Load Data (ALL Indirect Only samples and their spans)
#     comparison_data = load_creact_by_mention_type()
    
#     if not comparison_data:
#         print("ERROR: No comparison texts loaded. Check data paths and filtering.")
#         return

#     # 3. Run Detection and Get Results
#     confusion_matrix, race_keys, race_labels_official, MISCLASSIFICATION_CUE_TRACKER = run_bias_detection(
#         model, tokenizer, comparison_data, MISCLASSIFICATION_CUE_TRACKER
#     )

#     # 4. Print Classification Results (Confusion Matrix)
#     format_results_as_table(confusion_matrix, race_keys, race_labels_official)
    
#     # 5. Print Cue Analysis
#     print_cue_analysis(MISCLASSIFICATION_CUE_TRACKER)


# if __name__ == "__main__":
#     main()

# 3x3 activation analysi
# import torch
# import numpy as np
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from tqdm import tqdm
# import json
# import os
# import collections
# from modeling_qwen2_mlp import Qwen2Config, Qwen2ForCausalLM 

# # Register the custom model
# AutoModelForCausalLM.register(Qwen2Config, Qwen2ForCausalLM, exist_ok=True)

# # --- GLOBAL CONFIGURATION ---
# MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
# MAX_LENGTH = 512  
# # Data File Paths
# INDICATORS_PATH = 'indicators_df.jsonl'
# ASSIGNMENTS_PATH = 'all_re_assignments_df.jsonl'
# OUTPUT_FILE = 'misclassified_samples_with_activations_new.jsonl'  # Only misclassified samples

# # Define the target labels for the experiment (3 groups only)
# TARGET_RACES = {
#     'White': 'White',
#     'Black/AA': 'Black or African American',
#     'Asian': 'Asian'
# }

# # Define the C-REACT column names corresponding to the TARGET_RACES
# C_REACT_RACE_GROUPS = {
#     'White': 'RACE White',
#     'Black/AA': 'RACE Black or African American',
#     'Asian': 'RACE Asian',
# }

# # --- NEURON LIST FOR EXTRACTION ---
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
#     return model, tokenizer


# def load_creact_by_mention_type():
#     """
#     FIXED: Loads ALL C-REACT sentences (no sampling), filters to keep only 'Indirect Only' 
#     samples for the target races. Matches Script 1's data loading exactly.
#     """
#     print("\n--- Starting C-REACT Data Loading (All Indirect Samples) ---")
    
#     # 1. Load DataFrames
#     try:
#         df_indicators = pd.read_json(INDICATORS_PATH, lines=True)
#         df_assignments = pd.read_json(ASSIGNMENTS_PATH, lines=True)
#     except FileNotFoundError as e:
#         print(f"ERROR: Data file not found. Check paths: {e}")
#         return {}

#     # 2. Merge Data on sentence_id
#     df_merged = df_indicators.merge(df_assignments, on='sentence_id', suffixes=('_ind', '_assign'))
    
#     # --- 3. Classify Mentions at the Sentence Level ---
#     direct_labels = ['race', 'ethnicity']
#     indirect_labels = ['country', 'language']
    
#     def has_indicator_type(spans, types):
#         return any(span.get('label') in types for span in spans)

#     # FIXED: Handle potential column name differences after merge (matching Script 1)
#     if 'spans_ind' in df_merged.columns:
#         span_col = 'spans_ind'
#     else:
#         span_col = 'spans'

#     df_merged['is_direct'] = df_merged[span_col].apply(lambda s: has_indicator_type(s, direct_labels))
#     df_merged['is_indirect'] = df_merged[span_col].apply(lambda s: has_indicator_type(s, indirect_labels))
    
#     # 4. Determine Mention Group (Pure Groups Only)
#     df_merged['mention_type'] = 'Excluded'
    
#     # Pure Indirect Only: Must have indirect, must NOT have direct
#     df_merged.loc[~df_merged['is_direct'] & df_merged['is_indirect'], 'mention_type'] = 'Indirect Only'
    
#     # 5. Determine the Assigned Race Label
#     def get_assigned_race(row):
#         for simple_name, col_name in C_REACT_RACE_GROUPS.items():
#             if row.get(col_name) == 1:
#                 return simple_name
#         return None 

#     df_merged['assigned_race'] = df_merged.apply(get_assigned_race, axis=1)

#     # --- 6. Grouping (No Sampling Limit - Use ALL samples) ---
    
#     final_texts = collections.defaultdict(list)
    
#     # Filter for sentences with positive race labels AND 'Indirect Only' mention type
#     df_final_groups = df_merged[
#         df_merged['assigned_race'].notna() & 
#         (df_merged['mention_type'] == 'Indirect Only')
#     ].copy()

#     target_race_names = list(C_REACT_RACE_GROUPS.keys())
    
#     for race_name in target_race_names:
#         group_df = df_final_groups[df_final_groups['assigned_race'] == race_name]
        
#         group_key = f"{race_name}_Indirect Only"
        
#         # FIXED: No sampling - use all texts (matching Script 1)
#         texts = group_df['text_ind'].tolist()

#         if len(texts) > 0:
#              final_texts[group_key] = texts
    
#     print("\nTotal Indirect Samples loaded per race group:")
#     total_samples = 0
#     for group_name, texts in final_texts.items():
#         count = len(texts)
#         print(f"  {group_name}: {count} samples")
#         total_samples += count
        
#     print(f"Total samples for inference: {total_samples}")
    
#     return final_texts


# def run_bias_detection(model, tokenizer, comparison_texts, all_race_neurons):
#     """
#     Runs classification, collects all results, and extracts activations for all tracked neurons.
#     """
#     print("\n--- Starting Bias Detection, Classification, and Activation Extraction ---")
    
#     # Setup
#     race_keys = list(TARGET_RACES.keys())
#     race_labels_official = list(TARGET_RACES.values())
#     race_to_index = {race: i for i, race in enumerate(race_keys)}
#     prompt_options = ', '.join(race_labels_official)
    
#     all_results_list = []  # For saving - misclassified only
#     all_samples_for_activation = []  # For activation analysis - ALL samples
    
#     # Flatten all neurons into a single list
#     all_neurons_to_track_list = [
#         n for subdict in all_race_neurons.values() for sublist in subdict.values() for n in sublist
#     ]
    
#     # Initialize Confusion Matrix
#     matrix_size = len(race_keys)
#     confusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    
#     # Track sample count for diagnostic output
#     sample_count = 0

#     for group_key, texts in comparison_texts.items():
        
#         ground_truth_race = group_key.split('_')[0]
#         gt_index = race_to_index[ground_truth_race]
        
#         for text in tqdm(texts, desc=f"Processing {group_key} ({len(texts)} samples)"):
            
#             # FIXED: Use exact same prompt as Script 1
#             prompt = (
#     "Please read the following sentence from a clinical note. "
#     "Based on the information contained within the sentence, determine which of the following races or ethnicities the patient belongs to. "
#     "Please respond with only one option.\n\n"
#     f"Options: [{prompt_options}]\n\n"
#     f"Clinical Note: \"{text}\"\n\n"
#     "Inferred Race or Ethnicity:"
# )

            
#             # STEP 1: Generate classification answer
#             inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
            
#             with torch.no_grad():
#                 outputs_gen = model.generate(
#                     **inputs,
#                     max_new_tokens=10,  # FIXED: Match Script 1 (was 20)
#                     do_sample=False,
#                     pad_token_id=tokenizer.eos_token_id 
#                 )
            
#             predicted_answer_tokens = outputs_gen[0, inputs.input_ids.shape[-1]:] 
#             predicted_answer_raw = tokenizer.decode(predicted_answer_tokens, skip_special_tokens=True).strip()
            
#             # Diagnostic output for first 5 samples
#             sample_count += 1
#             if sample_count <= 5:
#                 print(f"\n=== SAMPLE {sample_count} DIAGNOSTIC ===")
#                 print(f"Ground Truth: {ground_truth_race}")
#                 print(f"Clinical Note (first 100 chars): {text[:100]}...")
#                 print(f"Model Output: {predicted_answer_raw}")
#                 print("="*50)
            
#             # Extract prediction
#             predicted_race_key = None
#             for race_key_test, race_label_test in TARGET_RACES.items():
#                 if race_label_test.lower() in predicted_answer_raw.lower() or race_key_test.lower() in predicted_answer_raw.lower():
#                     predicted_race_key = race_key_test
#                     break
            
#             is_correct = (predicted_race_key == ground_truth_race)
            
#             # Update Confusion Matrix
#             if predicted_race_key in race_to_index:
#                 pred_index = race_to_index[predicted_race_key]
#                 confusion_matrix[gt_index, pred_index] += 1
            
#             # STEP 2: Extract activations using a SEPARATE forward pass with same prompt
#             inputs_act = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
#             last_token_idx = inputs_act.input_ids.shape[1] - 1
            
#             activation_values = {}
            
#             try:
#                 with torch.no_grad():
#                     outputs_act = model(
#                         **inputs_act, 
#                         output_hidden_states=False, 
#                         output_mlp_pre_residual=True, 
#                         return_dict=True
#                     )
                
#                 all_mlp_activations = outputs_act.mlp_outputs 
                
#                 for layer_idx, neuron_idx in all_neurons_to_track_list:
#                     layer_activations = all_mlp_activations[layer_idx]
#                     act_value = layer_activations[0, last_token_idx, neuron_idx].item()
#                     activation_values[f"{layer_idx}_{neuron_idx}"] = act_value

#             except Exception as e:
#                 print(f"Error during activation extraction: {e}")
#                 activation_values = {}
            
#             # Save to activation analysis list (ALL samples)
#             if predicted_race_key is not None:
#                 all_samples_for_activation.append({
#                     'ground_truth': ground_truth_race,
#                     'predicted_race': predicted_race_key,
#                     'sentence_text': text,
#                     'model_output': predicted_answer_raw,
#                     'activations': activation_values
#                 })
            
#             # Save to results list (ONLY misclassified samples)
#             if not is_correct and predicted_race_key is not None:
#                 all_results_list.append({
#                     'ground_truth': ground_truth_race,
#                     'predicted_race': predicted_race_key,
#                     'sentence_text': text,
#                     'model_output': predicted_answer_raw,
#                     'activations': activation_values
#                 })
            
#             # Cleanup
#             del inputs, outputs_gen, inputs_act, outputs_act
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#     return all_results_list, all_samples_for_activation, confusion_matrix, race_keys, race_labels_official


# def analyze_and_print_activations(all_samples_data, all_race_neurons):
#     """
#     Groups ALL samples by (Actual Race, Predicted Race) and calculates 
#     the average activation for all six core neuron groups across all 3x3 conditions.
#     """
#     print("\n" + "="*120)
#     print("      DETAILED AVERAGED ACTIVATION ANALYSIS (Across All 3x3 Classification Outcomes)")
#     print("="*120)
    
#     # Prepare Neuron Group Mapping
#     neuron_map = {}
#     for race, mention_types in all_race_neurons.items():
#         for mention_type, neurons in mention_types.items():
#             group_name = f"{race} {mention_type}"
#             for layer, neuron in neurons:
#                 neuron_map[f"{layer}_{neuron}"] = group_name

#     grouped_activations = collections.defaultdict(lambda: collections.defaultdict(list))
    
#     race_keys = list(TARGET_RACES.keys())
    
#     # Collect Activations
#     for sample in all_samples_data:
#         gt_race = sample['ground_truth']
#         predicted_race = sample['predicted_race']
        
#         if gt_race not in race_keys or predicted_race not in race_keys:
#             continue
            
#         classification_key = f"{gt_race} -> {predicted_race}"
        
#         for neuron_id, activation_value in sample['activations'].items():
#             neuron_group_name = neuron_map.get(neuron_id)
            
#             if neuron_group_name:
#                 grouped_activations[classification_key][neuron_group_name].append(activation_value)

#     # Calculate averages
#     column_headers = []
#     for gt in race_keys:
#         for pred in race_keys:
#             column_headers.append(f"{gt} -> {pred}")

#     row_headers = sorted(list(set(neuron_map.values())))
    
#     data_matrix = collections.defaultdict(dict)

#     for neuron_group in row_headers:
#         for classification_key in column_headers:
#             acts = grouped_activations[classification_key].get(neuron_group, [])
#             avg_act = np.mean(acts) if acts else np.nan
#             data_matrix[neuron_group][classification_key] = avg_act

#     # Print Table
#     col_width = 13
#     header_width = 25
#     total_width = header_width + col_width * len(column_headers)
    
#     print("-" * total_width)
#     print(f"{'NEURON GROUP':<{header_width}} | {'Classification Outcome (Actual -> Predicted)':<{(col_width * len(column_headers) - 2)}}")
#     print("-" * total_width)
    
#     header_line = f"{'':<{header_width}} | " + "".join(f"{h:<{col_width}}" for h in column_headers)
#     print(header_line)
#     print("-" * total_width)

#     for neuron_group in row_headers:
#         row_values = []
#         for key in column_headers:
#             value = data_matrix[neuron_group][key]
#             formatted_value = f"{value:+.4f}" if not np.isnan(value) else "   N/A   "
#             row_values.append(formatted_value)
            
#         data_line = f"{neuron_group:<{header_width}} | " + "".join(f"{v:<{col_width}}" for v in row_values)
#         print(data_line)

#     print("=" * total_width)


# def format_results_as_table(matrix, race_keys, race_labels_official):
#     """Prints the numpy confusion matrix as a readable table."""
#     print("\n\n" + "="*80)
#     print("      MODEL CLASSIFICATION RESULTS (Indirect Context Bias)")
#     print("="*80)
    
#     key_to_official_label = dict(zip(race_keys, race_labels_official))
#     display_labels = [key_to_official_label[key] for key in race_keys]
    
#     header = ["Actual \\ Predicted"] + display_labels
#     row_format = "{:<20}" + "{:<15}" * len(display_labels)
    
#     print(row_format.format(*header))
#     print("-" * (20 + 15 * len(display_labels)))
    
#     total_accuracy = 0
#     total_samples = 0
    
#     for i, actual_race_key in enumerate(race_keys):
#         actual_race_display = key_to_official_label[actual_race_key]
#         row_data = [actual_race_display] + matrix[i, :].tolist()
        
#         row_total = np.sum(matrix[i, :])
#         correct_predictions = matrix[i, i]
        
#         print(row_format.format(*row_data[:len(display_labels) + 1]))

#         if row_total > 0:
#             total_accuracy += correct_predictions
#             total_samples += row_total
    
#     print("-" * (20 + 15 * len(display_labels)))
#     predicted_totals = np.sum(matrix, axis=0)
#     total_row = ["TOTAL PREDICTED"] + predicted_totals.tolist()
#     print(row_format.format(*total_row))

#     print("-" * (20 + 15 * len(display_labels)))
#     if total_samples > 0:
#         overall_accuracy = total_accuracy / total_samples
#         print(f"Overall Test Accuracy: {overall_accuracy:.4f} ({total_accuracy}/{total_samples})")
    
#     print("="*80)


# def main():
#     print("="*80)
#     print(f"   C-REACT Bias and Activation Analysis (ALL INDIRECT SAMPLES)")
#     print("="*80)
    
#     model, tokenizer = model_setup(MODEL_NAME)
    
#     # FIXED: No longer passes n_samples_per_group - uses all samples
#     comparison_texts = load_creact_by_mention_type()
    
#     if not comparison_texts:
#         print("ERROR: No comparison texts loaded. Check data paths and filtering.")
#         return

#     misclassified_samples, all_samples_for_activation, confusion_matrix, race_keys, race_labels_official = run_bias_detection(
#         model, 
#         tokenizer, 
#         comparison_texts, 
#         ALL_RACE_NEURONS
#     )
    
#     format_results_as_table(confusion_matrix, race_keys, race_labels_official)
    
#     # Use ALL samples for activation analysis (includes correct predictions)
#     analyze_and_print_activations(all_samples_for_activation, ALL_RACE_NEURONS)

#     # Save only misclassified samples to JSONL
#     with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#         for sample in misclassified_samples:
#             f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
#     print(f"\nAnalysis complete.")
#     print(f"  - Activation analysis performed on: {len(all_samples_for_activation)} total samples")
#     print(f"  - Misclassified samples saved to: {OUTPUT_FILE} ({len(misclassified_samples)} samples)")


# if __name__ == "__main__":
#     main()

# Cue + 3x3 combined analysis
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import os
import collections
# from modeling_qwen2_mlp import Qwen2Config, Qwen2ForCausalLM 
# AutoModelForCausalLM.register(Qwen2Config, Qwen2ForCausalLM, exist_ok=True)


MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.3'
MAX_LENGTH = 512

# Data File Paths
INDICATORS_PATH = 'indicators_df.jsonl'
ASSIGNMENTS_PATH = 'all_re_assignments_df.jsonl'
OUTPUT_FILE = 'misclassified_samples_with_activations_mistral.jsonl'

TARGET_RACES = {
    'White': 'White',
    'Black/AA': 'Black or African American',
    'Asian': 'Asian'
}

C_REACT_RACE_GROUPS = {
    'White': 'RACE White',
    'Black/AA': 'RACE Black or African American',
    'Asian': 'RACE Asian',
}

# Global Tracker for Cues: {'White -> Asian': {'language': {'Russain': 5}, ...}}
MISCLASSIFICATION_CUE_TRACKER = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))

ALL_RACE_NEURONS = {
    'Asian': {
        'Direct': [
            (31, 4453), (30, 2346), (29, 2137), (29, 8986),
        ],
    },
    'Black/AA': {
        'Direct': [
            (31, 5923), (30, 8715), (29, 3398), (29, 12572), (28, 13186),
        ],
        'Indirect': [
            (30, 8715),
        ]
    },
    'White': {
        'Direct': [
            (31, 1606), (31, 12760), (31, 9831), 
        ],
        'Indirect': [
            (31, 2399), (30, 7356), (29,4487),
            (28, 260),
        ]
    }
}

def model_setup(model_name):
    """Loads model and tokenizer."""
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

# def model_setup(model_name):
#     # Create a local cache folder that YOU own
#     my_local_cache = "./my_model_cache"
#     os.makedirs(my_local_cache, exist_ok=True)
    
#     print(f"Loading tokenizer and model from local cache: {my_local_cache}")

#     # 1. Load Tokenizer with cache_dir
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name,
#         cache_dir=my_local_cache  # <--- Critical Fix for Permission Error
#     )

#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     # 2. Load Model with cache_dir
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float32,
#         device_map="auto",
#         trust_remote_code=False,
#         cache_dir=my_local_cache  # <--- Critical Fix for Permission Error
#     )
    
#     model.eval()
#     print("Model loaded.")
#     return model, tokenizer

def load_creact_by_mention_type():
    """
    Loads ALL C-REACT sentences (no sampling), filters to keep only 'Indirect Only' 
    samples for the target races. Returns text AND span data for cue tracking.
    """
    print("\n--- Starting C-REACT Data Loading (All Indirect Samples) ---")
    

    df_indicators = pd.read_json(INDICATORS_PATH, lines=True)
    df_assignments = pd.read_json(ASSIGNMENTS_PATH, lines=True)

    df_merged = df_indicators.merge(df_assignments, on='sentence_id', suffixes=('_ind', '_assign'))
    
    direct_labels = ['race', 'ethnicity']
    indirect_labels = ['country', 'language']
    
    def has_indicator_type(spans, types):
        return any(span.get('label') in types for span in spans)

    if 'spans_ind' in df_merged.columns:
        span_col = 'spans_ind'
    else:
        span_col = 'spans'

    df_merged['is_direct'] = df_merged[span_col].apply(lambda s: has_indicator_type(s, direct_labels))
    df_merged['is_indirect'] = df_merged[span_col].apply(lambda s: has_indicator_type(s, indirect_labels))
    
    df_merged['mention_type'] = 'Excluded'
    
    # Pure Indirect Only
    df_merged.loc[~df_merged['is_direct'] & df_merged['is_indirect'], 'mention_type'] = 'Indirect Only'
    
    def get_assigned_race(row):
        for simple_name, col_name in C_REACT_RACE_GROUPS.items():
            if row.get(col_name) == 1:
                return simple_name
        return None 

    df_merged['assigned_race'] = df_merged.apply(get_assigned_race, axis=1)

    
    final_data = collections.defaultdict(list)
    
    # Filter for sentences with positive race labels and 'Indirect Only' mention type
    df_final_groups = df_merged[
        df_merged['assigned_race'].notna() & 
        (df_merged['mention_type'] == 'Indirect Only')
    ].copy()

    target_race_names = list(C_REACT_RACE_GROUPS.keys())
    
    for race_name in target_race_names:
        group_df = df_final_groups[df_final_groups['assigned_race'] == race_name]
        
        group_key = f"{race_name}_Indirect Only"
        
        # Store a dictionary with both text and spans for every sample (for cue tracking)
        samples = group_df[['text_ind', span_col]].rename(columns={span_col: 'spans'}).to_dict('records')

        if len(samples) > 0:
             final_data[group_key] = samples
    
    print("\nTotal Indirect Samples loaded per race group:")
    total_samples = 0
    for group_name, data in final_data.items():
        count = len(data)
        print(f"  {group_name}: {count} samples")
        total_samples += count
        
    print(f"Total samples for inference: {total_samples}")
    
    return final_data


def run_bias_detection(model, tokenizer, comparison_data, all_race_neurons, cue_tracker):
    """
    Runs classification, collects all results, extracts activations for all tracked neurons,
    and tracks cues for misclassifications.
    """
    print("\n--- Starting Bias Detection, Classification, Activation Extraction, and Cue Tracking ---")
    
    race_keys = list(TARGET_RACES.keys())
    race_labels_official = list(TARGET_RACES.values())
    race_to_index = {race: i for i, race in enumerate(race_keys)}
    prompt_options = ', '.join(race_labels_official)
    
    all_results_list = []  # For saving misclassified only
    all_samples_for_activation = []  # For activation analysis (all samples)
    

    all_neurons_to_track_list = [
        n for subdict in all_race_neurons.values() for sublist in subdict.values() for n in sublist
    ]
    
    matrix_size = len(race_keys)
    confusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    
    sample_count = 0

    for group_key, data_list in comparison_data.items():
        
        ground_truth_race = group_key.split('_')[0]
        gt_index = race_to_index[ground_truth_race]
        
        for sample in tqdm(data_list, desc=f"Processing {group_key} ({len(data_list)} samples)"):
            text = sample['text_ind']
            spans = sample['spans']
            
            prompt = (
    "Please read the following sentence from a clinical note. "
    "Based on the information contained within the sentence, determine which of the following races or ethnicities the patient belongs to. "
    "Please respond with only one option.\n\n"
    f"Options: [{prompt_options}]\n\n"
    f"Clinical Note: \"{text}\"\n\n"
    "Inferred Race or Ethnicity:"
)

            
            # Generate classification answer
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
            
            with torch.no_grad():
                outputs_gen = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id 
                )
            
            predicted_answer_tokens = outputs_gen[0, inputs.input_ids.shape[-1]:] 
            predicted_answer_raw = tokenizer.decode(predicted_answer_tokens, skip_special_tokens=True).strip()
            
            # Diagnostic output for first 5 samples
            sample_count += 1
            if sample_count <= 5:
                print(f"\n=== SAMPLE {sample_count} DIAGNOSTIC ===")
                print(f"Ground Truth: {ground_truth_race}")
                print(f"Clinical Note (first 100 chars): {text[:100]}...")
                print(f"Model Output: {predicted_answer_raw}")
                print("="*50)
            
            # Extract prediction
            predicted_race_key = None
            for race_key_test, race_label_test in TARGET_RACES.items():
                if race_label_test.lower() in predicted_answer_raw.lower() or race_key_test.lower() in predicted_answer_raw.lower():
                    predicted_race_key = race_key_test
                    break
            
            is_correct = (predicted_race_key == ground_truth_race)
            
            if predicted_race_key in race_to_index:
                pred_index = race_to_index[predicted_race_key]
                confusion_matrix[gt_index, pred_index] += 1
            
            # Extract activations using a separate forward pass with same prompt
            inputs_act = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
            last_token_idx = inputs_act.input_ids.shape[1] - 1
            
            activation_values = {}
            
            try:
                with torch.no_grad():
                    outputs_act = model(
                        **inputs_act, 
                        output_hidden_states=False, 
                        output_mlp_pre_residual=True, 
                        return_dict=True
                    )
                
                all_mlp_activations = outputs_act.mlp_outputs 
                
                for layer_idx, neuron_idx in all_neurons_to_track_list:
                    layer_activations = all_mlp_activations[layer_idx]
                    act_value = layer_activations[0, last_token_idx, neuron_idx].item()
                    activation_values[f"{layer_idx}_{neuron_idx}"] = act_value

            except Exception as e:
                print(f"Error during activation extraction: {e}")
                activation_values = {}
            
            # Save to activation analysis list (all samples)
            if predicted_race_key is not None:
                all_samples_for_activation.append({
                    'ground_truth': ground_truth_race,
                    'predicted_race': predicted_race_key,
                    'sentence_text': text,
                    'model_output': predicted_answer_raw,
                    'activations': activation_values
                })
            
            # Save only misclassified samples
            if not is_correct and predicted_race_key is not None:
                all_results_list.append({
                    'ground_truth': ground_truth_race,
                    'predicted_race': predicted_race_key,
                    'sentence_text': text,
                    'model_output': predicted_answer_raw,
                    'activations': activation_values
                })
                
                # Track cues for misclassifications
                misclass_key = f"{ground_truth_race} -> {predicted_race_key}"
                
                for span in spans:
                    label = span.get('label')
                
                    start = span.get('start')
                    end = span.get('end')
                    span_text = text[start:end].strip() if start is not None and end is not None else ''

                    if label in ['country', 'language'] and span_text:
                        cue_tracker[misclass_key][label][span_text] += 1
            
            del inputs, outputs_gen, inputs_act, outputs_act
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_results_list, all_samples_for_activation, confusion_matrix, race_keys, race_labels_official, cue_tracker


def analyze_and_print_activations(all_samples_data, all_race_neurons):
    """
    Groups ALL samples by (Actual Race, Predicted Race) and calculates 
    the average activation for all six core neuron groups across all 3x3 conditions.
    """
    print("\n" + "="*120)
    print("      DETAILED AVERAGED ACTIVATION ANALYSIS (Across All 3x3 Classification Outcomes)")
    print("="*120)
    
    # Prepare Neuron Group Mapping
    neuron_map = {}
    for race, mention_types in all_race_neurons.items():
        for mention_type, neurons in mention_types.items():
            group_name = f"{race} {mention_type}"
            for layer, neuron in neurons:
                neuron_map[f"{layer}_{neuron}"] = group_name

    grouped_activations = collections.defaultdict(lambda: collections.defaultdict(list))
    
    race_keys = list(TARGET_RACES.keys())
    
    # Collect Activations
    for sample in all_samples_data:
        gt_race = sample['ground_truth']
        predicted_race = sample['predicted_race']
        
        if gt_race not in race_keys or predicted_race not in race_keys:
            continue
            
        classification_key = f"{gt_race} -> {predicted_race}"
        
        for neuron_id, activation_value in sample['activations'].items():
            neuron_group_name = neuron_map.get(neuron_id)
            
            if neuron_group_name:
                grouped_activations[classification_key][neuron_group_name].append(activation_value)

    # Calculate averages
    column_headers = []
    for gt in race_keys:
        for pred in race_keys:
            column_headers.append(f"{gt} -> {pred}")

    row_headers = sorted(list(set(neuron_map.values())))
    
    data_matrix = collections.defaultdict(dict)

    for neuron_group in row_headers:
        for classification_key in column_headers:
            acts = grouped_activations[classification_key].get(neuron_group, [])
            avg_act = np.mean(acts) if acts else np.nan
            data_matrix[neuron_group][classification_key] = avg_act

    col_width = 13
    header_width = 25
    total_width = header_width + col_width * len(column_headers)
    
    print("-" * total_width)
    print(f"{'NEURON GROUP':<{header_width}} | {'Classification Outcome (Actual -> Predicted)':<{(col_width * len(column_headers) - 2)}}")
    print("-" * total_width)
    
    header_line = f"{'':<{header_width}} | " + "".join(f"{h:<{col_width}}" for h in column_headers)
    print(header_line)
    print("-" * total_width)

    for neuron_group in row_headers:
        row_values = []
        for key in column_headers:
            value = data_matrix[neuron_group][key]
            formatted_value = f"{value:+.4f}" if not np.isnan(value) else "   N/A   "
            row_values.append(formatted_value)
            
        data_line = f"{neuron_group:<{header_width}} | " + "".join(f"{v:<{col_width}}" for v in row_values)
        print(data_line)

    print("=" * total_width)


def format_results_as_table(matrix, race_keys, race_labels_official):
    """Prints the numpy confusion matrix."""
    print("\n\n" + "="*80)
    print("      MODEL CLASSIFICATION RESULTS (Indirect Context Bias)")
    print("="*80)
    
    key_to_official_label = dict(zip(race_keys, race_labels_official))
    display_labels = [key_to_official_label[key] for key in race_keys]
    
    header = ["Actual \\ Predicted"] + display_labels
    row_format = "{:<20}" + "{:<15}" * len(display_labels)
    
    print(row_format.format(*header))
    print("-" * (20 + 15 * len(display_labels)))
    
    total_accuracy = 0
    total_samples = 0
    
    for i, actual_race_key in enumerate(race_keys):
        actual_race_display = key_to_official_label[actual_race_key]
        row_data = [actual_race_display] + matrix[i, :].tolist()
        
        row_total = np.sum(matrix[i, :])
        correct_predictions = matrix[i, i]
        
        print(row_format.format(*row_data[:len(display_labels) + 1]))

        if row_total > 0:
            total_accuracy += correct_predictions
            total_samples += row_total
    
    print("-" * (20 + 15 * len(display_labels)))
    predicted_totals = np.sum(matrix, axis=0)
    total_row = ["TOTAL PREDICTED"] + predicted_totals.tolist()
    print(row_format.format(*total_row))

    print("-" * (20 + 15 * len(display_labels)))
    if total_samples > 0:
        overall_accuracy = total_accuracy / total_samples
        print(f"Overall Test Accuracy: {overall_accuracy:.4f} ({total_accuracy}/{total_samples})")
    
    print("="*80)


def print_cue_analysis(cue_tracker):
    """Prints the final statistics on misclassification cues."""
    print("\n\n" + "="*80)
    print("      MISCLASSIFICATION CUE ANALYSIS (What triggered the error?)")
    print("="*80)
    
    sorted_misclass_keys = sorted(cue_tracker.keys())
    
    if not sorted_misclass_keys:
        print("No misclassifications recorded (or no cues found in misclassified samples).")
    
    for misclass_key in sorted_misclass_keys:
        cue_data = cue_tracker[misclass_key]
        print(f"\n--- Misclassification Type: {misclass_key} ---")
        
        if 'language' in cue_data:
            print("  [LANGUAGE Cues]")
            sorted_cues = sorted(cue_data['language'].items(), key=lambda x: x[1], reverse=True)
            for cue, count in sorted_cues:
                print(f"    - {cue}: {count}")
        if 'country' in cue_data:
            print("  [COUNTRY Cues]")
            sorted_cues = sorted(cue_data['country'].items(), key=lambda x: x[1], reverse=True)
            for cue, count in sorted_cues:
                print(f"    - {cue}: {count}")

    print("="*80)


def main():
    print("="*80)
    print(f"   C-REACT Bias and Activation Analysis (ALL INDIRECT SAMPLES)")
    print("="*80)
    
    global MISCLASSIFICATION_CUE_TRACKER
    
    model, tokenizer = model_setup(MODEL_NAME)
    
    comparison_data = load_creact_by_mention_type()
    
    if not comparison_data:
        print("ERROR: No comparison texts loaded. Check data paths and filtering.")
        return

    misclassified_samples, all_samples_for_activation, confusion_matrix, race_keys, race_labels_official, MISCLASSIFICATION_CUE_TRACKER = run_bias_detection(
        model, 
        tokenizer, 
        comparison_data, 
        ALL_RACE_NEURONS,
        MISCLASSIFICATION_CUE_TRACKER
    )
    
    # Print Classification Results (Confusion Matrix)
    format_results_as_table(confusion_matrix, race_keys, race_labels_official)
    
    analyze_and_print_activations(all_samples_for_activation, ALL_RACE_NEURONS)
    
    # Print Cue Analysis
    print_cue_analysis(MISCLASSIFICATION_CUE_TRACKER)

    # Save only misclassified samples to JSONL
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for sample in misclassified_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
    print(f"\nAnalysis complete.")
    print(f"  - Activation analysis performed on: {len(all_samples_for_activation)} total samples")
    print(f"  - Misclassified samples saved to: {OUTPUT_FILE} ({len(misclassified_samples)} samples)")


if __name__ == "__main__":
    main()

# #collect all activations for wrong and correct classifications only for llama and mistra;
# import torch
# import numpy as np
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from tqdm import tqdm
# import json
# import os
# import collections

# # --- GLOBAL CONFIGURATION ---
# # Choose one of these models:
# # MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
# # MODEL_NAME = 'meta-llama/Llama-3.1-8B-Instruct'
# # MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.3'
# MODEL_NAME = 'meta-llama/Llama-3.1-8B-Instruct'  # Change this to switch models

# MAX_LENGTH = 512

# # Data File Paths
# INDICATORS_PATH = 'indicators_df.jsonl'
# ASSIGNMENTS_PATH = 'all_re_assignments_df.jsonl'
# OUTPUT_FILE = 'misclassified_samples_with_activations_llama3_new.jsonl'  # Only misclassified samples

# # Define the target labels for the experiment (3 groups only)
# TARGET_RACES = {
#     'White': 'White',
#     'Black/AA': 'Black or African American',
#     'Asian': 'Asian'
# }

# # Define the C-REACT column names corresponding to the TARGET_RACES
# C_REACT_RACE_GROUPS = {
#     'White': 'RACE White',
#     'Black/AA': 'RACE Black or African American',
#     'Asian': 'RACE Asian',
# }

# # Global Tracker for Cues: {'White -> Asian': {'language': {'Spanish': 5}, ...}}
# MISCLASSIFICATION_CUE_TRACKER = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))

# # --- NEURON LIST FOR EXTRACTION ---
# ALL_RACE_NEURONS = {
#     'Asian': {
#         'Direct': [
#             (31, 5691), (30, 14299), (28, 5272),
#         ],
#         'Indirect': [
#             (31, 5691), (31, 6950), (28, 10616),
#         ]
#     },
#     'Black/AA': {
#         'Direct': [
#             (29, 7195), (29, 3868), (28, 13826), (28, 6824),
#         ],
#         'Indirect': [
#             (29, 7195), (28, 6824), (28, 13826),
#         ]
#     },
#     'White': {
#         'Direct': [
#             (30, 9094), 
#         ],
#         'Indirect': [
#             (31, 10606), (30, 12584), (31,10409),
#             (28, 4193),
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
#     print(f"Hidden size: {model.config.hidden_size}")
#     print(f"Intermediate size: {model.config.intermediate_size}")
    
#     # Detect model architecture for MLP extraction
#     model_type = model.config.model_type
#     print(f"Model type detected: {model_type}")
    
#     return model, tokenizer, model_type

# # def model_setup(model_name):
# #     # Create a local cache folder that YOU own
# #     my_local_cache = "./my_model_cache"
# #     os.makedirs(my_local_cache, exist_ok=True)
    
# #     print(f"Loading tokenizer and model from local cache: {my_local_cache}")

# #     # 1. Load Tokenizer with cache_dir
# #     tokenizer = AutoTokenizer.from_pretrained(
# #         model_name,
# #         cache_dir=my_local_cache  # <--- Critical Fix for Permission Error
# #     )

# #     if tokenizer.pad_token is None:
# #         tokenizer.pad_token = tokenizer.eos_token

# #     # 2. Load Model with cache_dir
# #     model = AutoModelForCausalLM.from_pretrained(
# #         model_name,
# #         torch_dtype=torch.float32,
# #         device_map="auto",
# #         cache_dir=my_local_cache  # <--- Critical Fix for Permission Error
# #     )
    
# #     model.eval()
# #     print(f"Model loaded. Layers: {model.config.num_hidden_layers}")
# #     print(f"Hidden size: {model.config.hidden_size}")
# #     print(f"Intermediate size: {model.config.intermediate_size}")
    
# #     # Detect model architecture for MLP extraction
# #     model_type = model.config.model_type
# #     print(f"Model type detected: {model_type}")

# #     return model, tokenizer, model_type


# def get_mlp_module(model, layer_idx, model_type):
#     """
#     Returns the MLP output module for a given layer based on model architecture.
#     Different models have different naming conventions.
#     """
#     if model_type == 'llama':
#         # Llama: model.layers[i].mlp.down_proj
#         return model.model.layers[layer_idx].mlp.down_proj
#     elif model_type == 'mistral':
#         # Mistral: model.layers[i].mlp.down_proj
#         return model.model.layers[layer_idx].mlp.down_proj
#     elif model_type == 'qwen2':
#         # Qwen2: model.layers[i].mlp.down_proj
#         return model.model.layers[layer_idx].mlp.down_proj
#     else:
#         raise ValueError(f"Unsupported model type: {model_type}")


# def extract_mlp_activations(model, inputs, neurons_to_track, model_type, debug=False):
#     """
#     Extract MLP activations for specified neurons using forward hooks.
    
#     We hook the INPUT to down_proj, which gives us the intermediate MLP activations
#     (after up_proj/gate_proj and activation function, before down_proj).
#     This is where the "neuron" activations live (intermediate_size dimension).
    
#     Args:
#         model: The language model
#         inputs: Tokenized inputs
#         neurons_to_track: List of (layer_idx, neuron_idx) tuples
#         model_type: String indicating model architecture
#         debug: If True, print debug information
    
#     Returns:
#         Dictionary mapping 'layer_neuron' to activation value at last token position
#     """
#     activation_values = {}
#     hooks = []
#     mlp_outputs = {}
    
#     # Get unique layers we need to hook
#     layers_needed = set(layer_idx for layer_idx, _ in neurons_to_track)
    
#     if debug:
#         print(f"  [DEBUG] Layers to hook: {layers_needed}")
#         print(f"  [DEBUG] Neurons to track: {neurons_to_track[:5]}...")  # Show first 5
    
#     def make_hook(layer_idx):
#         def hook(module, input, output):
#             # input is a tuple, input[0] is the actual tensor
#             # Shape: (batch_size, seq_len, intermediate_size)
#             if len(input) > 0 and input[0] is not None:
#                 mlp_outputs[layer_idx] = input[0].detach()
#                 if debug and layer_idx == list(layers_needed)[0]:  # Debug first layer only
#                     print(f"  [DEBUG] Layer {layer_idx} hook fired!")
#                     print(f"  [DEBUG] Input shape: {input[0].shape}")
#         return hook
    
#     # Register hooks for each layer we need
#     for layer_idx in layers_needed:
#         try:
#             mlp_module = get_mlp_module(model, layer_idx, model_type)
#             hook = mlp_module.register_forward_hook(make_hook(layer_idx))
#             hooks.append(hook)
#             if debug:
#                 print(f"  [DEBUG] Registered hook for layer {layer_idx}")
#         except Exception as e:
#             print(f"  [ERROR] Failed to register hook for layer {layer_idx}: {e}")
    
#     try:
#         # Forward pass
#         with torch.no_grad():
#             model(**inputs)
        
#         if debug:
#             print(f"  [DEBUG] Forward pass complete. Captured layers: {list(mlp_outputs.keys())}")
        
#         # Extract activations at last token position
#         last_token_idx = inputs.input_ids.shape[1] - 1
        
#         for layer_idx, neuron_idx in neurons_to_track:
#             if layer_idx in mlp_outputs:
#                 tensor = mlp_outputs[layer_idx]
#                 if neuron_idx < tensor.shape[2]:
#                     act_value = tensor[0, last_token_idx, neuron_idx].item()
#                     activation_values[f"{layer_idx}_{neuron_idx}"] = act_value
#                 elif debug:
#                     print(f"  [ERROR] Neuron {neuron_idx} out of range for layer {layer_idx} (max: {tensor.shape[2]-1})")
#             elif debug:
#                 print(f"  [ERROR] Layer {layer_idx} not in captured outputs")
    
#     finally:
#         # Remove all hooks
#         for hook in hooks:
#             hook.remove()
    
#     if debug:
#         print(f"  [DEBUG] Extracted {len(activation_values)} activation values")
#         # Print all activation values for first sample
#         print(f"  [DEBUG] Activation values:")
#         for neuron_id, act_val in activation_values.items():
#             print(f"    Neuron {neuron_id}: {act_val:+.6f}")
    
#     return activation_values


# def load_creact_by_mention_type():
#     """
#     Loads ALL C-REACT sentences (no sampling), filters to keep only 'Indirect Only' 
#     samples for the target races. Returns text AND span data for cue tracking.
#     """
#     print("\n--- Starting C-REACT Data Loading (All Indirect Samples) ---")
    
#     # 1. Load DataFrames
#     try:
#         df_indicators = pd.read_json(INDICATORS_PATH, lines=True)
#         df_assignments = pd.read_json(ASSIGNMENTS_PATH, lines=True)
#     except FileNotFoundError as e:
#         print(f"ERROR: Data file not found. Check paths: {e}")
#         return {}

#     # 2. Merge Data on sentence_id
#     df_merged = df_indicators.merge(df_assignments, on='sentence_id', suffixes=('_ind', '_assign'))
    
#     # --- 3. Classify Mentions at the Sentence Level ---
#     direct_labels = ['race', 'ethnicity']
#     indirect_labels = ['country', 'language']
    
#     def has_indicator_type(spans, types):
#         return any(span.get('label') in types for span in spans)

#     # Handle potential column name differences after merge
#     if 'spans_ind' in df_merged.columns:
#         span_col = 'spans_ind'
#     else:
#         span_col = 'spans'

#     df_merged['is_direct'] = df_merged[span_col].apply(lambda s: has_indicator_type(s, direct_labels))
#     df_merged['is_indirect'] = df_merged[span_col].apply(lambda s: has_indicator_type(s, indirect_labels))
    
#     # 4. Determine Mention Group (Pure Groups Only)
#     df_merged['mention_type'] = 'Excluded'
    
#     # Pure Indirect Only: Must have indirect, must NOT have direct
#     df_merged.loc[~df_merged['is_direct'] & df_merged['is_indirect'], 'mention_type'] = 'Indirect Only'
    
#     # 5. Determine the Assigned Race Label
#     def get_assigned_race(row):
#         for simple_name, col_name in C_REACT_RACE_GROUPS.items():
#             if row.get(col_name) == 1:
#                 return simple_name
#         return None 

#     df_merged['assigned_race'] = df_merged.apply(get_assigned_race, axis=1)

#     # --- 6. Grouping (No Sampling Limit - Use ALL samples) ---
    
#     final_data = collections.defaultdict(list)
    
#     # Filter for sentences with positive race labels AND 'Indirect Only' mention type
#     df_final_groups = df_merged[
#         df_merged['assigned_race'].notna() & 
#         (df_merged['mention_type'] == 'Indirect Only')
#     ].copy()

#     target_race_names = list(C_REACT_RACE_GROUPS.keys())
    
#     for race_name in target_race_names:
#         group_df = df_final_groups[df_final_groups['assigned_race'] == race_name]
        
#         group_key = f"{race_name}_Indirect Only"
        
#         # Store a dictionary with both text and spans for every sample (for cue tracking)
#         samples = group_df[['text_ind', span_col]].rename(columns={span_col: 'spans'}).to_dict('records')

#         if len(samples) > 0:
#              final_data[group_key] = samples
    
#     print("\nTotal Indirect Samples loaded per race group:")
#     total_samples = 0
#     for group_name, data in final_data.items():
#         count = len(data)
#         print(f"  {group_name}: {count} samples")
#         total_samples += count
        
#     print(f"Total samples for inference: {total_samples}")
    
#     return final_data


# def run_bias_detection(model, tokenizer, comparison_data, all_race_neurons, cue_tracker, model_type):
#     """
#     Runs classification, collects all results, extracts activations for all tracked neurons,
#     and tracks cues for misclassifications.
#     """
#     print("\n--- Starting Bias Detection, Classification, Activation Extraction, and Cue Tracking ---")
    
#     # Setup
#     race_keys = list(TARGET_RACES.keys())
#     race_labels_official = list(TARGET_RACES.values())
#     race_to_index = {race: i for i, race in enumerate(race_keys)}
#     prompt_options = ', '.join(race_labels_official)
    
#     all_results_list = []  # For saving - misclassified only
#     all_samples_for_activation = []  # For activation analysis - ALL samples
    
#     # Flatten all neurons into a single list
#     all_neurons_to_track_list = [
#         n for subdict in all_race_neurons.values() for sublist in subdict.values() for n in sublist
#     ]
    
#     # Initialize Confusion Matrix
#     matrix_size = len(race_keys)
#     confusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    
#     # Track sample count for diagnostic output
#     sample_count = 0

#     for group_key, data_list in comparison_data.items():
        
#         ground_truth_race = group_key.split('_')[0]
#         gt_index = race_to_index[ground_truth_race]
        
#         for sample in tqdm(data_list, desc=f"Processing {group_key} ({len(data_list)} samples)"):
#             text = sample['text_ind']
#             spans = sample['spans']
            
#             # Use exact same prompt as Script 1
#             prompt = (
#     "Please read the following sentence from a clinical note. "
#     "Based on the information contained within the sentence, determine which of the following races or ethnicities the patient belongs to. "
#     "Please respond with only one option.\n\n"
#     f"Options: [{prompt_options}]\n\n"
#     f"Clinical Note: \"{text}\"\n\n"
#     "Inferred Race or Ethnicity:"
# )

            
#             # STEP 1: Generate classification answer
#             inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
            
#             with torch.no_grad():
#                 outputs_gen = model.generate(
#                     **inputs,
#                     max_new_tokens=10,
#                     do_sample=False,
#                     pad_token_id=tokenizer.eos_token_id 
#                 )
            
#             predicted_answer_tokens = outputs_gen[0, inputs.input_ids.shape[-1]:] 
#             predicted_answer_raw = tokenizer.decode(predicted_answer_tokens, skip_special_tokens=True).strip()
            
#             # Diagnostic output for first 5 samples
#             sample_count += 1
#             if sample_count <= 5:
#                 print(f"\n=== SAMPLE {sample_count} DIAGNOSTIC ===")
#                 print(f"Ground Truth: {ground_truth_race}")
#                 print(f"Clinical Note (first 100 chars): {text[:100]}...")
#                 print(f"Model Output: {predicted_answer_raw}")
#                 print("="*50)
            
#             # Extract prediction
#             predicted_race_key = None
#             for race_key_test, race_label_test in TARGET_RACES.items():
#                 if race_label_test.lower() in predicted_answer_raw.lower() or race_key_test.lower() in predicted_answer_raw.lower():
#                     predicted_race_key = race_key_test
#                     break
            
#             is_correct = (predicted_race_key == ground_truth_race)
            
#             # Update Confusion Matrix
#             if predicted_race_key in race_to_index:
#                 pred_index = race_to_index[predicted_race_key]
#                 confusion_matrix[gt_index, pred_index] += 1
            
#             # STEP 2: Extract activations using hooks
#             inputs_act = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
            
#             try:
#                 # Enable debug for first 5 samples
#                 debug_mode = (sample_count <= 5)
#                 activation_values = extract_mlp_activations(
#                     model, inputs_act, all_neurons_to_track_list, model_type, debug=(sample_count == 1)
#                 )
                
#                 # Print activation values for first 5 samples
#                 if debug_mode and activation_values:
#                     print(f"\n  [ACTIVATIONS for Sample {sample_count}]")
#                     for neuron_id, act_val in list(activation_values.items())[:10]:  # Show first 10
#                         print(f"    Neuron {neuron_id}: {act_val:+.4f}")
#                     if len(activation_values) > 10:
#                         print(f"    ... and {len(activation_values) - 10} more neurons")
#                 elif debug_mode and not activation_values:
#                     print(f"\n  [WARNING] No activations captured for Sample {sample_count}!")
                    
#             except Exception as e:
#                 print(f"Error during activation extraction: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 activation_values = {}
            
#             # Save to activation analysis list (ALL samples)
#             if predicted_race_key is not None:
#                 all_samples_for_activation.append({
#                     'ground_truth': ground_truth_race,
#                     'predicted_race': predicted_race_key,
#                     'sentence_text': text,
#                     'model_output': predicted_answer_raw,
#                     'activations': activation_values
#                 })
            
#             # Save to results list (ONLY misclassified samples)
#             if not is_correct and predicted_race_key is not None:
#                 all_results_list.append({
#                     'ground_truth': ground_truth_race,
#                     'predicted_race': predicted_race_key,
#                     'sentence_text': text,
#                     'model_output': predicted_answer_raw,
#                     'activations': activation_values
#                 })
                
#                 # STEP 3: Track cues for misclassifications
#                 misclass_key = f"{ground_truth_race} -> {predicted_race_key}"
                
#                 for span in spans:
#                     label = span.get('label')
                    
#                     # Extract text using start/end offsets
#                     start = span.get('start')
#                     end = span.get('end')
#                     span_text = text[start:end].strip() if start is not None and end is not None else ''
                    
#                     # We are interested in 'country' and 'language' indicators
#                     if label in ['country', 'language'] and span_text:
#                         cue_tracker[misclass_key][label][span_text] += 1
            
#             # Cleanup
#             del inputs, outputs_gen, inputs_act
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#     return all_results_list, all_samples_for_activation, confusion_matrix, race_keys, race_labels_official, cue_tracker


# def analyze_and_print_activations(all_samples_data, all_race_neurons):
#     """
#     Groups ALL samples by (Actual Race, Predicted Race) and calculates 
#     the average activation for all six core neuron groups across all 3x3 conditions.
#     """
#     print("\n" + "="*120)
#     print("      DETAILED AVERAGED ACTIVATION ANALYSIS (Across All 3x3 Classification Outcomes)")
#     print("="*120)
    
#     # Prepare Neuron Group Mapping
#     neuron_map = {}
#     for race, mention_types in all_race_neurons.items():
#         for mention_type, neurons in mention_types.items():
#             group_name = f"{race} {mention_type}"
#             for layer, neuron in neurons:
#                 neuron_map[f"{layer}_{neuron}"] = group_name

#     grouped_activations = collections.defaultdict(lambda: collections.defaultdict(list))
    
#     race_keys = list(TARGET_RACES.keys())
    
#     # Collect Activations
#     for sample in all_samples_data:
#         gt_race = sample['ground_truth']
#         predicted_race = sample['predicted_race']
        
#         if gt_race not in race_keys or predicted_race not in race_keys:
#             continue
            
#         classification_key = f"{gt_race} -> {predicted_race}"
        
#         for neuron_id, activation_value in sample['activations'].items():
#             neuron_group_name = neuron_map.get(neuron_id)
            
#             if neuron_group_name:
#                 grouped_activations[classification_key][neuron_group_name].append(activation_value)

#     # Calculate averages
#     column_headers = []
#     for gt in race_keys:
#         for pred in race_keys:
#             column_headers.append(f"{gt} -> {pred}")

#     row_headers = sorted(list(set(neuron_map.values())))
    
#     data_matrix = collections.defaultdict(dict)

#     for neuron_group in row_headers:
#         for classification_key in column_headers:
#             acts = grouped_activations[classification_key].get(neuron_group, [])
#             avg_act = np.mean(acts) if acts else np.nan
#             data_matrix[neuron_group][classification_key] = avg_act

#     # Print Table
#     col_width = 13
#     header_width = 25
#     total_width = header_width + col_width * len(column_headers)
    
#     print("-" * total_width)
#     print(f"{'NEURON GROUP':<{header_width}} | {'Classification Outcome (Actual -> Predicted)':<{(col_width * len(column_headers) - 2)}}")
#     print("-" * total_width)
    
#     header_line = f"{'':<{header_width}} | " + "".join(f"{h:<{col_width}}" for h in column_headers)
#     print(header_line)
#     print("-" * total_width)

#     for neuron_group in row_headers:
#         row_values = []
#         for key in column_headers:
#             value = data_matrix[neuron_group][key]
#             formatted_value = f"{value:+.4f}" if not np.isnan(value) else "   N/A   "
#             row_values.append(formatted_value)
            
#         data_line = f"{neuron_group:<{header_width}} | " + "".join(f"{v:<{col_width}}" for v in row_values)
#         print(data_line)

#     print("=" * total_width)


# def format_results_as_table(matrix, race_keys, race_labels_official):
#     """Prints the numpy confusion matrix as a readable table."""
#     print("\n\n" + "="*80)
#     print("      MODEL CLASSIFICATION RESULTS (Indirect Context Bias)")
#     print("="*80)
    
#     key_to_official_label = dict(zip(race_keys, race_labels_official))
#     display_labels = [key_to_official_label[key] for key in race_keys]
    
#     header = ["Actual \\ Predicted"] + display_labels
#     row_format = "{:<20}" + "{:<15}" * len(display_labels)
    
#     print(row_format.format(*header))
#     print("-" * (20 + 15 * len(display_labels)))
    
#     total_accuracy = 0
#     total_samples = 0
    
#     for i, actual_race_key in enumerate(race_keys):
#         actual_race_display = key_to_official_label[actual_race_key]
#         row_data = [actual_race_display] + matrix[i, :].tolist()
        
#         row_total = np.sum(matrix[i, :])
#         correct_predictions = matrix[i, i]
        
#         print(row_format.format(*row_data[:len(display_labels) + 1]))

#         if row_total > 0:
#             total_accuracy += correct_predictions
#             total_samples += row_total
    
#     print("-" * (20 + 15 * len(display_labels)))
#     predicted_totals = np.sum(matrix, axis=0)
#     total_row = ["TOTAL PREDICTED"] + predicted_totals.tolist()
#     print(row_format.format(*total_row))

#     print("-" * (20 + 15 * len(display_labels)))
#     if total_samples > 0:
#         overall_accuracy = total_accuracy / total_samples
#         print(f"Overall Test Accuracy: {overall_accuracy:.4f} ({total_accuracy}/{total_samples})")
    
#     print("="*80)


# def print_cue_analysis(cue_tracker):
#     """Prints the final statistics on misclassification cues."""
#     print("\n\n" + "="*80)
#     print("      MISCLASSIFICATION CUE ANALYSIS (What triggered the error?)")
#     print("="*80)
    
#     # Sort keys for consistent printing
#     sorted_misclass_keys = sorted(cue_tracker.keys())
    
#     if not sorted_misclass_keys:
#         print("No misclassifications recorded (or no cues found in misclassified samples).")
    
#     for misclass_key in sorted_misclass_keys:
#         cue_data = cue_tracker[misclass_key]
#         print(f"\n--- Misclassification Type: {misclass_key} ---")
        
#         # Check Language
#         if 'language' in cue_data:
#             print("  [LANGUAGE Cues]")
#             sorted_cues = sorted(cue_data['language'].items(), key=lambda x: x[1], reverse=True)
#             for cue, count in sorted_cues:
#                 print(f"    - {cue}: {count}")
        
#         # Check Country
#         if 'country' in cue_data:
#             print("  [COUNTRY Cues]")
#             sorted_cues = sorted(cue_data['country'].items(), key=lambda x: x[1], reverse=True)
#             for cue, count in sorted_cues:
#                 print(f"    - {cue}: {count}")

#     print("="*80)


# def main():
#     print("="*80)
#     print(f"   C-REACT Bias and Activation Analysis (ALL INDIRECT SAMPLES)")
#     print(f"   Model: {MODEL_NAME}")
#     print("="*80)
    
#     global MISCLASSIFICATION_CUE_TRACKER
    
#     model, tokenizer, model_type = model_setup(MODEL_NAME)
    
#     # Load data with spans for cue tracking
#     comparison_data = load_creact_by_mention_type()
    
#     if not comparison_data:
#         print("ERROR: No comparison texts loaded. Check data paths and filtering.")
#         return

#     misclassified_samples, all_samples_for_activation, confusion_matrix, race_keys, race_labels_official, MISCLASSIFICATION_CUE_TRACKER = run_bias_detection(
#         model, 
#         tokenizer, 
#         comparison_data, 
#         ALL_RACE_NEURONS,
#         MISCLASSIFICATION_CUE_TRACKER,
#         model_type
#     )
    
#     # Print Classification Results (Confusion Matrix)
#     format_results_as_table(confusion_matrix, race_keys, race_labels_official)
    
#     # Use ALL samples for activation analysis (includes correct predictions)
#     analyze_and_print_activations(all_samples_for_activation, ALL_RACE_NEURONS)
    
#     # Print Cue Analysis
#     print_cue_analysis(MISCLASSIFICATION_CUE_TRACKER)

#     # Save only misclassified samples to JSONL
#     with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#         for sample in misclassified_samples:
#             f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
#     print(f"\nAnalysis complete.")
#     print(f"  - Activation analysis performed on: {len(all_samples_for_activation)} total samples")
#     print(f"  - Misclassified samples saved to: {OUTPUT_FILE} ({len(misclassified_samples)} samples)")


# if __name__ == "__main__":
#     main()