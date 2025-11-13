"""
Activation extraction for GSM8K with exact lm_eval replication and reasoning-focused features.

Key improvements:
1. Separate forward pass with output_hidden_states=True (fixes KV cache issue)
2. Reasoning token identification (tokens between prompt and "####")
3. Reasoning-focused pooled features for linear probing
4. Full verification against lm_eval results

Features extracted per layer:
- reasoning_last: last reasoning token
- reasoning_mean: mean over all reasoning tokens
- reasoning_prefix_X_last: last token in first X% of reasoning (X=25,50,75)
- reasoning_prefix_X_mean: mean over first X% of reasoning (X=25,50,75)
- last_token, mean_pooled, max_pooled: reference features over full sequence
"""
import json
import re
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList


class MultiTokenStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria that checks for multiple stop strings in the decoded output.
    This matches lm_eval's behavior for the 'until' parameter.
    """
    def __init__(self, stop_strings: List[str], tokenizer, prompt_length: int):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Decode only the generated part (after prompt)
        generated_ids = input_ids[0][self.prompt_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Check if any stop string appears in the generated text
        for stop_string in self.stop_strings:
            if stop_string in generated_text:
                return True
        return False


def find_reasoning_token_span(prompt: str, generated_text: str, full_text: str, 
                              input_ids: torch.Tensor) -> tuple:
    """
    Find the token span corresponding to reasoning tokens.
    Reasoning tokens are between the end of the prompt and the "####" marker.
    
    Args:
        prompt: The original prompt string
        generated_text: The generated response text (after prompt)
        full_text: The full concatenated text (prompt + generated_text)
        input_ids: The tokenized sequence
        
    Returns:
        (reasoning_start_idx, reasoning_end_idx): Token indices for reasoning span
    """
    # Find where "####" appears in the generated text
    if "####" in generated_text:
        answer_start_in_generated = generated_text.find("####")
        reasoning_text = generated_text[:answer_start_in_generated]
    else:
        # If no ####, consider all generated text as reasoning
        reasoning_text = generated_text
    
    # The reasoning starts right after the prompt
    reasoning_start_char = len(prompt)
    reasoning_end_char = len(prompt) + len(reasoning_text)
    
    # Return as token indices (approximate - we'll use prompt token count)
    # reasoning_start_idx is the length of the prompt in tokens
    # reasoning_end_idx needs to be computed from the full sequence
    return reasoning_start_char, reasoning_end_char


def compute_reasoning_features(hidden_states: torch.Tensor, 
                               reasoning_start_idx: int, 
                               reasoning_end_idx: int) -> Dict:
    """
    Compute pooled features over reasoning tokens.
    
    Args:
        hidden_states: Shape (seq_len, hidden_dim) - hidden states for one layer
        reasoning_start_idx: Start token index for reasoning
        reasoning_end_idx: End token index for reasoning (exclusive)
        
    Returns:
        Dictionary of pooled features
    """
    features = {}
    
    # Extract reasoning tokens only
    reasoning_hidden = hidden_states[reasoning_start_idx:reasoning_end_idx]
    reasoning_len = reasoning_hidden.shape[0]
    
    if reasoning_len == 0:
        # No reasoning tokens found, return zeros
        hidden_dim = hidden_states.shape[1]
        zero_vec = torch.zeros(hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device)
        return {
            'reasoning_last': zero_vec.float().cpu().numpy().tolist(),
            'reasoning_mean': zero_vec.float().cpu().numpy().tolist(),
            'reasoning_prefix_25_last': zero_vec.float().cpu().numpy().tolist(),
            'reasoning_prefix_50_last': zero_vec.float().cpu().numpy().tolist(),
            'reasoning_prefix_75_last': zero_vec.float().cpu().numpy().tolist(),
            'reasoning_prefix_25_mean': zero_vec.float().cpu().numpy().tolist(),
            'reasoning_prefix_50_mean': zero_vec.float().cpu().numpy().tolist(),
            'reasoning_prefix_75_mean': zero_vec.float().cpu().numpy().tolist(),
        }
    
    # Full reasoning features (convert bfloat16 to float32 for numpy compatibility)
    features['reasoning_last'] = reasoning_hidden[-1].float().cpu().numpy().tolist()
    features['reasoning_mean'] = reasoning_hidden.mean(dim=0).float().cpu().numpy().tolist()
    
    # Prefix features (25%, 50%, 75%)
    for prefix_pct in [25, 50, 75]:
        prefix_idx = max(1, int(reasoning_len * prefix_pct / 100))
        prefix_hidden = reasoning_hidden[:prefix_idx]
        
        features[f'reasoning_prefix_{prefix_pct}_last'] = prefix_hidden[-1].float().cpu().numpy().tolist()
        features[f'reasoning_prefix_{prefix_pct}_mean'] = prefix_hidden.mean(dim=0).float().cpu().numpy().tolist()
    
    return features


def extract_answer_gsm8k(text: str, fallback: str = "[invalid]") -> str:
    """
    Extract the final numerical answer from GSM8K response.
    Uses the same regex pattern and logic as lm_eval's "strict-match" filter.
    
    Args:
        text: The model's generated response
        fallback: Value to return if no match found
        
    Returns:
        Extracted answer string or fallback
    """
    # Same regex pattern used in lm_eval for GSM8K
    regex_pattern = r"#### (\-?[0-9\.\,]+)"
    regex = re.compile(regex_pattern)
    
    match = regex.findall(text)
    if match:
        # Take the first match (equivalent to "take_first" filter)
        answer = match[0].strip()
        return answer
    else:
        return fallback


def evaluate_exact_match_gsm8k(prediction: str, reference: str) -> bool:
    """
    Evaluate exact match using the same logic as lm_eval for GSM8K.
    
    Applies the following transformations:
    - Remove commas
    - Remove dollar signs
    - Remove "#### " prefix
    - Remove trailing periods
    - Compare case-sensitively (ignore_case=True but numbers don't have case)
    
    Args:
        prediction: The extracted answer from model response
        reference: The target answer from the dataset
        
    Returns:
        True if answers match, False otherwise
    """
    # If prediction is invalid, it's not a match
    if prediction == "[invalid]":
        return False
    
    # Apply the same regexes_to_ignore as defined in gsm8k.yaml
    regexes_to_ignore = [
        r",",           # Remove commas
        r"\$",          # Remove dollar signs  
        r"(?s).*#### ", # Remove everything before and including "#### "
        r"\.$"          # Remove trailing period
    ]
    
    pred = prediction
    ref = reference
    
    for pattern in regexes_to_ignore:
        pred = re.sub(pattern, "", pred)
        ref = re.sub(pattern, "", ref)
    
    # Strip whitespace
    pred = pred.strip()
    ref = ref.strip()
    
    # Case insensitive comparison (though for numbers it doesn't matter)
    pred = pred.lower()
    ref = ref.lower()
    
    return pred == ref


def extract_activations_from_lm_eval_output(
    samples_jsonl_path: str,
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    layer_indices: List[int] = None,
    output_path: str = "outputs/activations/activations_with_results.json",
    device: str = "cuda:0",
    max_samples: int = None,
):
    """
    Extract activations using the exact prompts from lm_eval output.
    
    Args:
        samples_jsonl_path: Path to the samples JSONL file from lm_eval
        model_name: HuggingFace model name
        layer_indices: Specific layer indices to extract (e.g., [10, 15, 20, 25])
        output_path: Where to save the activation data
        device: Device to run on
        max_samples: Maximum number of samples to process (None = all)
    """
    
    print(f"Loading model: {model_name}")
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    
    print(f"Model loaded on {device}")
    
    # Determine which layers to extract
    if layer_indices is None:
        num_layers = len(model.model.layers)
        layer_indices = list(range(num_layers))
    print(f"Will extract from layers: {layer_indices}")
    
    # Load samples from lm_eval output
    print(f"Loading samples from: {samples_jsonl_path}")
    samples = []
    with open(samples_jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"Processing {len(samples)} samples...")
    
    activation_data = []
    missing_exact_match_count = 0  # Track samples without exact_match field
    skipped_mismatch_count = 0     # Track samples skipped due to inference mismatch
    processed_count = 0             # Track successfully processed samples
    
    for sample in tqdm(samples, desc="Extracting activations"):
        doc_id = sample['doc_id']
        
        # Get the EXACT prompt used by lm_eval
        prompt = sample['arguments']['gen_args_0']['arg_0']
        
        # Get generation kwargs
        gen_kwargs = sample['arguments']['gen_args_0']['arg_1']
        
        # Get the evaluation results
        if 'exact_match' not in sample:
            missing_exact_match_count += 1
            is_correct = False  # Default for missing field
        else:
            is_correct = sample.get('exact_match', 0) == 1.0
        model_response = sample['resps'][0][0] if sample['resps'] else ""
        
        # Tokenize the prompt (exactly as lm_eval does)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Run inference to capture activations
        # Use the EXACT generation parameters from lm_eval
        gen_params = {
            'max_new_tokens': 512,
            'do_sample': gen_kwargs.get('do_sample', False),
            'pad_token_id': tokenizer.eos_token_id,
            'return_dict_in_generate': True,
            'output_hidden_states': False,  # We're using hooks instead
        }
        
        # Add temperature only if sampling
        if gen_params['do_sample']:
            gen_params['temperature'] = gen_kwargs.get('temperature', 1.0)
        
        # CRITICAL: Add the stopping sequences (until parameter) to match lm_eval exactly
        if 'until' in gen_kwargs and gen_kwargs['until']:
            stop_strings = gen_kwargs['until']
            stopping_criteria = StoppingCriteriaList([
                MultiTokenStoppingCriteria(
                    stop_strings=stop_strings,
                    tokenizer=tokenizer,
                    prompt_length=inputs.input_ids.shape[1]
                )
            ])
            gen_params['stopping_criteria'] = stopping_criteria
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_params
            )
        
        # Decode the generated response (only the generated part, not the prompt)
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Truncate at stop strings (same as lm_eval does)
        if 'until' in gen_kwargs and gen_kwargs['until']:
            for stop_string in gen_kwargs['until']:
                if stop_string in generated_text:
                    generated_text = generated_text.split(stop_string)[0]
                    break
        
        # Extract the answer from our generated response using the same logic as lm_eval
        our_extracted_answer = extract_answer_gsm8k(generated_text)
        
        # Get the original lm_eval extracted answer
        lm_eval_extracted_answer = sample['filtered_resps'][0] if sample.get('filtered_resps') else "[invalid]"
        
        # Evaluate our response using the same logic as lm_eval
        target_answer = sample['target']
        our_is_correct = evaluate_exact_match_gsm8k(our_extracted_answer, target_answer)
        
        # Check if our result matches the original lm_eval result
        # Compare both the extracted answer AND the correctness evaluation
        answers_match = (our_extracted_answer == lm_eval_extracted_answer)
        
        # Also verify correctness matches (if exact_match field exists)
        if 'exact_match' in sample:
            lm_eval_is_correct = sample['exact_match'] == 1.0
            correctness_matches = (our_is_correct == lm_eval_is_correct)
        else:
            # If no exact_match field, we can't verify correctness
            correctness_matches = True  # Don't skip based on this
        
        # Skip this sample if there's a mismatch (different inference results)
        if not answers_match or not correctness_matches:
            skipped_mismatch_count += 1
            print(f"\n⚠️  Skipping doc_id {doc_id}: Inference mismatch detected")
            print(f"   Our answer: {our_extracted_answer}, LM-eval answer: {lm_eval_extracted_answer}")
            print(f"   Our correct: {our_is_correct}, LM-eval correct: {lm_eval_is_correct if 'exact_match' in sample else 'N/A'}")
            continue  # Skip this sample
        
        processed_count += 1
        
        # NOW: Extract activation features with a separate forward pass
        # Construct the full text (prompt + generated response)
        full_text = prompt + generated_text
        
        # Tokenize the full sequence
        full_inputs = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_token_length = inputs.input_ids.shape[1]
        
        # Run a forward pass to get hidden states for all layers
        with torch.no_grad():
            outputs = model(
                **full_inputs,
                output_hidden_states=True,
                use_cache=False
            )
        
        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Each tensor has shape (batch_size, seq_len, hidden_dim)
        # Index 0 is embeddings, indices 1..num_layers are layer outputs
        hidden_states_all_layers = outputs.hidden_states
        
        # Find reasoning token span
        # Reasoning tokens start after the prompt and end before "####"
        reasoning_start_idx = prompt_token_length
        
        # Find where "####" appears in the generated text to mark end of reasoning
        if "####" in generated_text:
            # Tokenize up to "####" to find the token index
            text_before_answer = prompt + generated_text.split("####")[0]
            tokens_before_answer = tokenizer(text_before_answer, return_tensors="pt").input_ids
            reasoning_end_idx = tokens_before_answer.shape[1]
        else:
            # If no ####, use the full sequence
            reasoning_end_idx = full_inputs.input_ids.shape[1]
        
        # Extract features for each requested layer
        sample_activations = {}
        
        for layer_idx in layer_indices:
            # hidden_states_all_layers[0] is embeddings
            # hidden_states_all_layers[layer_idx + 1] is the output of layer_idx
            layer_hidden = hidden_states_all_layers[layer_idx + 1][0]  # Remove batch dim
            
            # Compute reasoning-based features
            reasoning_features = compute_reasoning_features(
                layer_hidden, 
                reasoning_start_idx, 
                reasoning_end_idx
            )
            
            # Also keep some traditional features for reference (convert bfloat16 to float32)
            last_token_act = layer_hidden[-1].float().cpu().numpy().tolist()
            mean_act = layer_hidden.mean(dim=0).float().cpu().numpy().tolist()
            max_act = layer_hidden.max(dim=0)[0].float().cpu().numpy().tolist()
            
            sample_activations[f"layer_{layer_idx}"] = {
                **reasoning_features,  # Primary features
                'last_token': last_token_act,  # For reference
                'mean_pooled': mean_act,  # For reference
                'max_pooled': max_act,  # For reference
            }
        
        # Compile all data for this sample
        # Note: We use our verified inference results, not the original sample's
        activation_data.append({
            'doc_id': doc_id,
            'question': sample['doc']['question'],
            'target': sample['target'],
            'prompt': prompt,  # The exact prompt used
            'model_response': generated_text,  # Our generated response (verified to match)
            'extracted_answer': our_extracted_answer,  # The extracted numerical answer
            'is_correct': our_is_correct,  # Our evaluation (verified to match lm_eval)
            'activations': sample_activations,
            # Metadata about token spans (useful for debugging)
            'reasoning_token_span': {
                'start': reasoning_start_idx,
                'end': reasoning_end_idx,
                'length': reasoning_end_idx - reasoning_start_idx,
                'prompt_length': prompt_token_length,
                'full_length': full_inputs.input_ids.shape[1]
            },
            # Include original lm_eval metadata
            'doc_hash': sample.get('doc_hash'),
            'prompt_hash': sample.get('prompt_hash'),
            'target_hash': sample.get('target_hash'),
        })
    
    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(activation_data, f, indent=2)
    
    print(f"\nSaved activations to: {output_file}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"EXTRACTION STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples attempted: {len(samples)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped due to inference mismatch: {skipped_mismatch_count}")
    print(f"Samples missing 'exact_match' field: {missing_exact_match_count}")
    
    if len(activation_data) > 0:
        correct_count = sum(1 for item in activation_data if item['is_correct'])
        accuracy = correct_count / len(activation_data)
        print(f"\nAccuracy (of processed samples): {accuracy:.2%} ({correct_count}/{len(activation_data)})")
    
    print(f"Layers extracted: {layer_indices}")
    print(f"{'='*60}\n")
    
    return activation_data


def verify_mapping(activation_file: str, samples_jsonl_path: str):
    """
    Verify that activations are correctly mapped to lm_eval results.
    This now verifies that:
    1. The extracted answers match the lm_eval filtered responses
    2. The correctness evaluations match
    3. All samples in the activation file passed the mismatch check
    """
    # Load both files
    with open(activation_file, 'r') as f:
        activation_data = json.load(f)
    
    samples = []
    with open(samples_jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Create lookup by doc_id
    samples_dict = {s['doc_id']: s for s in samples}
    
    # Verify mapping
    print("\n" + "="*60)
    print("VERIFYING ACTIVATION MAPPING")
    print("="*60)
    all_match = True
    
    # Check all samples (or first 10 for quick verification)
    samples_to_check = min(10, len(activation_data))
    
    for act_item in activation_data[:samples_to_check]:
        doc_id = act_item['doc_id']
        sample = samples_dict[doc_id]
        
        # Verify extracted answer matches
        lm_eval_answer = sample.get('filtered_resps', [[None]])[0]
        our_answer = act_item.get('extracted_answer')
        
        # Verify correctness matches
        expected_correct = sample.get('exact_match', 0) == 1.0
        actual_correct = act_item['is_correct']
        
        answer_match = (our_answer == lm_eval_answer) if lm_eval_answer else True
        correct_match = (actual_correct == expected_correct)
        
        if not answer_match or not correct_match:
            print(f"❌ Mismatch at doc_id {doc_id}")
            print(f"   Answer: {our_answer} vs {lm_eval_answer}")
            print(f"   Correct: {actual_correct} vs {expected_correct}")
            all_match = False
        else:
            print(f"✓ doc_id {doc_id}: answer={our_answer}, correct={actual_correct}")
    
    if all_match:
        print("\n✅ All mappings verified correctly!")
        print(f"   Checked {samples_to_check} samples from {len(activation_data)} total")
    else:
        print("\n⚠️ Some mappings don't match!")
    
    print("="*60 + "\n")
    
    return all_match


def calculate_layer_indices(num_layers: int) -> List[int]:
    """
    Calculate layer indices to probe based on total number of layers.
    Returns layers at: early, early-mid, middle, middle-to-last, last positions.
    
    Args:
        num_layers: Total number of layers in the model
        
    Returns:
        List of layer indices to extract
    """
    # Define positions as percentages
    positions = {
        'early': 0.15,           # 15% through the network
        'early_mid': 0.35,       # 35% through
        'middle': 0.50,          # 50% through (middle)
        'mid_late': 0.70,        # 70% through
        'late': 0.90,            # 90% through (near end)
    }
    
    layer_indices = []
    for name, percentage in positions.items():
        layer_idx = int(num_layers * percentage)
        # Ensure we don't exceed bounds
        layer_idx = min(layer_idx, num_layers - 1)
        layer_indices.append(layer_idx)
    
    # Remove duplicates and sort
    layer_indices = sorted(list(set(layer_indices)))
    
    print(f"Auto-calculated layer indices for {num_layers}-layer model:")
    position_names = ['early', 'early_mid', 'middle', 'mid_late', 'late']
    for i, idx in enumerate(layer_indices[:len(position_names)]):
        percentage = (idx / num_layers) * 100
        print(f"  {position_names[i]:12s}: layer {idx:2d} ({percentage:.1f}%)")
    
    return layer_indices


if __name__ == "__main__":
    # Extract activations from your existing lm_eval output
   
    samples_path = "outputs/qwen25-math-gsm8k/Qwen__Qwen2.5-Math-1.5B/samples_gsm8k_2025-11-11T07-12-57.068782.jsonl"
    model_name = "Qwen/Qwen2.5-Math-1.5B"
    
    # Load model temporarily to get number of layers
    print(f"Loading model to determine layer count: {model_name}")
    temp_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    num_layers = len(temp_model.model.layers)
    print(f"Model has {num_layers} layers\n")
    
    # Calculate which layers to extract
    layers_to_extract = calculate_layer_indices(num_layers)
    
    # Clean up temporary model
    del temp_model
    torch.cuda.empty_cache()
    
    activation_data = extract_activations_from_lm_eval_output(
        samples_jsonl_path=samples_path,
        model_name=model_name,
        layer_indices=layers_to_extract,
        output_path="outputs/activations/activations_gsm8k.json",
        device="cuda:0",
        max_samples=2,  # Start with small number for testing, set to None for all
    )
    
    # Verify the mapping is correct
    verify_mapping(
        "outputs/activations/activations_gsm8k.json",
        samples_path
    )