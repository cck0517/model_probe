"""
Entropy baseline for GSM8K and hendrycks_math datasets.

This script computes the mean entropy over generated tokens as a baseline
for predicting answer correctness. Uses the same model and dataset as
get_activations.py but only collects entropy statistics.

Output: JSON file with each task's mean entropy and correctness.
"""
import json
import re
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

# Import helper functions from get_activations
from get_activations import (
    MultiTokenStoppingCriteria,
    extract_answer_gsm8k,
    extract_answer_hendrycks_math,
    evaluate_exact_match_gsm8k,
    evaluate_exact_match_hendrycks_math,
    is_equiv,
)


def compute_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy for each token position from logits.
    
    Args:
        logits: Shape (seq_len, vocab_size) - logits for each token position
        
    Returns:
        Tensor of shape (seq_len,) - entropy for each position
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Compute entropy: H = -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum(dim=-1)
    
    return entropy


def extract_entropy_from_lm_eval_output(
    samples_jsonl_path: str,
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    output_path: str = "outputs/entropy/entropy_with_results.json",
    device: str = "cuda:0",
    max_samples: int = None,
    dataset_type: str = "gsm8k",  # "gsm8k" or "hendrycks_math"
):
    """
    Extract mean entropy for each inference using the exact prompts from lm_eval output.
    
    Args:
        samples_jsonl_path: Path to the samples JSONL file from lm_eval
        model_name: HuggingFace model name
        output_path: Where to save the entropy data
        device: Device to run on
        max_samples: Maximum number of samples to process (None = all)
        dataset_type: "gsm8k" or "hendrycks_math" - determines answer extraction logic
    """
    
    print(f"Loading model: {model_name}")
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    
    print(f"Model loaded on {device}")
    
    # Load samples from lm_eval output
    print(f"Loading samples from: {samples_jsonl_path}")
    samples = []
    with open(samples_jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    if max_samples:
        samples = samples[:max_samples]
        print(f"Limiting to {max_samples} samples")
    
    print(f"Processing {len(samples)} samples...")
    
    entropy_data = []
    missing_exact_match_count = 0
    skipped_mismatch_count = 0
    processed_count = 0
    
    for sample in tqdm(samples, desc="Computing entropy"):
        doc_id = sample['doc_id']
        
        # Get the EXACT prompt used by lm_eval
        prompt = sample['arguments']['gen_args_0']['arg_0']
        
        # Get generation kwargs
        gen_kwargs = sample['arguments']['gen_args_0']['arg_1']
        
        # Get the evaluation results from lm_eval
        if 'exact_match' not in sample:
            missing_exact_match_count += 1
            lm_eval_is_correct = False  # Default for missing field
        else:
            lm_eval_is_correct = sample.get('exact_match', 0) == 1.0
        
        # Tokenize the prompt (exactly as lm_eval does)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_token_length = inputs.input_ids.shape[1]
        
        # Run inference with output_scores to get logits for entropy computation
        # Use the EXACT generation parameters from lm_eval
        gen_params = {
            'max_new_tokens': 512,
            'do_sample': gen_kwargs.get('do_sample', False),
            'pad_token_id': tokenizer.eos_token_id,
            'return_dict_in_generate': True,
            'output_scores': True,  # Get logits for entropy computation
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
                    prompt_length=prompt_token_length
                )
            ])
            gen_params['stopping_criteria'] = stopping_criteria
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_params
            )
        
        # Decode the generated response (only the generated part, not the prompt)
        generated_ids = outputs.sequences[0][prompt_token_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Truncate at stop strings (same as lm_eval does)
        if 'until' in gen_kwargs and gen_kwargs['until']:
            for stop_string in gen_kwargs['until']:
                if stop_string in generated_text:
                    generated_text = generated_text.split(stop_string)[0]
                    break
        
        # Extract the answer from our generated response using the same logic as lm_eval
        if dataset_type == "hendrycks_math":
            our_extracted_answer = extract_answer_hendrycks_math(generated_text)
            lm_eval_response = sample['filtered_resps'][0] if sample.get('filtered_resps') else ""
            lm_eval_extracted_answer = extract_answer_hendrycks_math(lm_eval_response)
            target_answer = sample['target']
            our_is_correct = evaluate_exact_match_hendrycks_math(our_extracted_answer, target_answer)
        else:  # gsm8k
            our_extracted_answer = extract_answer_gsm8k(generated_text)
            lm_eval_extracted_answer = sample['filtered_resps'][0] if sample.get('filtered_resps') else "[invalid]"
            target_answer = sample['target']
            our_is_correct = evaluate_exact_match_gsm8k(our_extracted_answer, target_answer)
        
        # Check if our result matches the original lm_eval result
        if dataset_type == "hendrycks_math":
            answers_match = is_equiv(our_extracted_answer, lm_eval_extracted_answer) if lm_eval_extracted_answer != "[invalid]" else False
        else:
            answers_match = (our_extracted_answer == lm_eval_extracted_answer)
        
        # Also verify correctness matches (if exact_match field exists)
        if 'exact_match' in sample:
            correctness_matches = (our_is_correct == lm_eval_is_correct)
        else:
            correctness_matches = True  # Don't skip based on this
        
        # Skip this sample if there's a mismatch (different inference results)
        if not answers_match or not correctness_matches:
            skipped_mismatch_count += 1
            print(f"\n⚠️  Skipping doc_id {doc_id}: Inference mismatch detected")
            print(f"   Our answer: {our_extracted_answer}, LM-eval answer: {lm_eval_extracted_answer}")
            print(f"   Our correct: {our_is_correct}, LM-eval correct: {lm_eval_is_correct if 'exact_match' in sample else 'N/A'}")
            continue  # Skip this sample
        
        processed_count += 1
        
        # Compute entropy from logits
        # outputs.scores is a tuple of (num_generated_tokens,) tensors
        # Each tensor has shape (batch_size, vocab_size)
        if outputs.scores and len(outputs.scores) > 0:
            # Stack all logits: (num_generated_tokens, vocab_size)
            all_logits = torch.stack(outputs.scores, dim=0).squeeze(1)  # Remove batch dim
            
            # Compute entropy for each generated token
            token_entropies = compute_entropy_from_logits(all_logits)
            
            # Compute mean entropy over all generated tokens
            mean_entropy = token_entropies.mean().item()
            
            # Also store per-token entropies for analysis (optional, can be large)
            # Convert to list for JSON serialization
            token_entropies_list = token_entropies.cpu().numpy().tolist()
        else:
            # No tokens generated (shouldn't happen, but handle gracefully)
            mean_entropy = 0.0
            token_entropies_list = []
            print(f"⚠️  Warning: No scores for doc_id {doc_id}")
        
        # Handle different field names for different datasets
        if dataset_type == "hendrycks_math":
            question_field = sample['doc'].get('problem', '')
        else:  # gsm8k
            question_field = sample['doc'].get('question', '')
        
        # Compile data for this sample
        entropy_data.append({
            'doc_id': doc_id,
            'question': question_field,
            'target': sample['target'],
            'prompt': prompt,
            'model_response': generated_text,
            'extracted_answer': our_extracted_answer,
            'is_correct': our_is_correct,
            'mean_entropy': mean_entropy,
            'num_generated_tokens': len(token_entropies_list),
            # Include original lm_eval metadata
            'doc_hash': sample.get('doc_hash'),
            'prompt_hash': sample.get('prompt_hash'),
            'target_hash': sample.get('target_hash'),
        })
    
    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(entropy_data, f, indent=2)
    
    print(f"\nSaved entropy data to: {output_file}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"ENTROPY EXTRACTION STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples attempted: {len(samples)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped due to inference mismatch: {skipped_mismatch_count}")
    print(f"Samples missing 'exact_match' field: {missing_exact_match_count}")
    
    if len(entropy_data) > 0:
        correct_count = sum(1 for item in entropy_data if item['is_correct'])
        accuracy = correct_count / len(entropy_data)
        print(f"\nAccuracy (of processed samples): {accuracy:.2%} ({correct_count}/{len(entropy_data)})")
        
        # Entropy statistics
        correct_entropies = [item['mean_entropy'] for item in entropy_data if item['is_correct']]
        incorrect_entropies = [item['mean_entropy'] for item in entropy_data if not item['is_correct']]
        
        if correct_entropies:
            print(f"\nMean entropy (correct): {np.mean(correct_entropies):.4f} ± {np.std(correct_entropies):.4f}")
        if incorrect_entropies:
            print(f"Mean entropy (incorrect): {np.mean(incorrect_entropies):.4f} ± {np.std(incorrect_entropies):.4f}")
    
    print(f"{'='*60}\n")
    
    return entropy_data


if __name__ == "__main__":
    # Extract entropy for Qwen 2.5 Math 1.5B on GSM8K
    samples_path = "outputs/qwen25-math-gsm8k/Qwen__Qwen2.5-Math-1.5B/samples_gsm8k_2025-11-11T07-12-57.068782.jsonl"
    model_name = "Qwen/Qwen2.5-Math-1.5B"
    
    entropy_data = extract_entropy_from_lm_eval_output(
        samples_jsonl_path=samples_path,
        model_name=model_name,
        output_path="outputs/entropy/entropy_qwen25_math_gsm8k.json",
        device="cuda:0",
        max_samples=500,  # Control number of tasks to test
        dataset_type="gsm8k",
    )

