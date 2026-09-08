#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KV-Cached Memory Agent + HotpotQA Dataset Evaluation

Adapted from GAM Framework to use KV-Cached Memory Agent System
"""

import gc
import json
import logging
import os
import re
import string
from collections import Counter

# Add project root to path
from pathlib import Path
from typing import Any, Dict, List, Optional

import tiktoken
import torch
from openai import OpenAI
from tqdm import tqdm

from config import ensure_app_config, load_raw_config
from evaluation.config_utils import resolve_eval_config_path
from src.conversation_manager.factory import create_chat_manager

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = "evaluation/hotpotqa/config.yaml"


# hotpotqa needs more careful management of cuda memory, in case of OOM errors
def force_cleanup_gpu_memory():
    """
    Force cleanup of GPU memory more aggressively.
    This includes clearing transformers caches and CUDA contexts.
    """
    # 1. Clear Python garbage first
    gc.collect()
    gc.collect()
    gc.collect()
    
    # 2. Clear transformers internal caches if available
    try:
        import transformers
        if hasattr(transformers, 'utils') and hasattr(transformers.utils, 'hub'):
            # Clear download cache references (won't delete files)
            pass
    except Exception:
        pass
    
    # 3. Clear any sentence_transformers cache
    try:
        pass
        # SentenceTransformer uses its own cache
    except Exception:
        pass
    
    # 4. CUDA cleanup
    if torch.cuda.is_available():
        # Synchronize all streams
        torch.cuda.synchronize()
        
        # Empty cache
        torch.cuda.empty_cache()
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Additional: try to reset accumulated state
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

# ========== Logging Setup ==========

def setup_logger(log_file: str) -> logging.Logger:
    """Setup logging configuration."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
    
    logger = logging.getLogger('hotpotqa_eval')
    return logger


# ========== Data Loading ==========

def load_hotpotqa(json_path: str) -> List[Dict[str, Any]]:
    """Load HotpotQA JSON dataset"""
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    data_all = [
        {
            "index": item.get("index", idx),
            "context": item.get("context", ""),
            "input": item.get("input", ""),
            "answers": item.get("answers", []),
            "_id": f"hotpotqa-{item.get('index', idx)}"
        }
        for idx, item in enumerate(dataset)
    ]
    
    return data_all


# ========== Context Splitting ==========

def build_context_chunks_for_sample(
    sample: Dict[str, Any], 
    max_tokens: int = 2000, 
    logger: Optional[logging.Logger] = None
) -> List[str]:
    """Split context text into chunks based on token count"""
    if logger is None:
        logger = logging.getLogger('hotpotqa_eval')
        
    context_text = sample.get("context") or ""
    
    if not context_text:
        return []
    
    tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    tokens = tokenizer.encode(context_text, disallowed_special=())
    
    if len(tokens) <= max_tokens:
        return [context_text]
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens)
        
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        
        start_idx = end_idx
    
    return chunks


# ========== Prompt Design ==========

def make_prompt(context: str, question: str) -> str:
    """Create unified prompt (open QA format)"""
    prompt = f"""You are an expert at answering questions concisely based on the provided context.

Instructions:
1.  **Analyze the Request**: Identify the specific entity, date, number, or name requested.
2.  **Reasoning**: Think step-by-step to locate the answer in the context. If the question asks for a former name, time-specific detail, or multiple entities, ensure you select the exact one matching the criteria.
3.  **Extraction**: Extract ONLY the specific answer string. Remove all articles (a, an, the), punctuation, and sentence structures.
4. **Note**: ALWAYS DIRECTLY OUTPUT the SHORT ANSWER not a sentence.



Question: {question}

Context: {context}
"""
    return prompt


# ========== Answer Evaluation ==========

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def _calculate_f1(pred_answer: str, gold_answers: List[str]) -> float:
    max_f1 = 0.0
    for gold_answer in gold_answers:
        max_f1 = max(max_f1, qa_f1_score(pred_answer, gold_answer))
    return max_f1


# ========== Core Processing Logic ==========

def process_sample(
    sample: Dict[str, Any], 
    sample_index: int, 
    outdir: str,
    working_api_key: str,
    working_base_url: str,
    working_model: str,
    config: dict,
    max_tokens: int = 2000,
    logger: Optional[logging.Logger] = None
):
    """Process a single sample using KV-Cached Memory Agent"""
    if logger is None:
        logger = logging.getLogger('hotpotqa_eval')

    app_config = ensure_app_config(config)
        
    sample_id = sample.get("_id", f"sample-{sample_index}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing sample #{sample_index}: {sample_id}")
    logger.info(f"{'='*60}")
    
    agent = None  # Initialize agent to None for proper cleanup in finally block
    
    try:
        # 1. Build context chunks
        context_chunks = build_context_chunks_for_sample(sample, max_tokens, logger)
        logger.info(f"Number of context chunks: {len(context_chunks)}")
        if context_chunks:
            logger.info(f"First context chunk preview:\n{context_chunks[0][:400]}...")
        
        # Create output directory
        sample_results_dir = os.path.join(outdir, sample_id)
        os.makedirs(sample_results_dir, exist_ok=True)
        logger.info(f"Output directory: {sample_results_dir}")
        
        # 2. Create ChatManager with clean cache
        logger.info("\nStep 1: Create ChatManager")
        runtime_kwargs = app_config.to_chat_manager_kwargs()
        runtime_kwargs["clean_cache_first"] = True
        agent = create_chat_manager(**runtime_kwargs)
        logger.info("[OK] ChatManager created")
        
        # 3. Add context chunks to memory
        logger.info("\nStep 2: Add context chunks to memory")
        for i, context_chunk in enumerate(context_chunks, 1):
            logger.info(f"  Processing context chunk {i}/{len(context_chunks)}...")
            agent.add_memory(context_chunk)
        logger.info(f"[OK] Memory building completed! Added {len(context_chunks)} chunks")
        
        # 4. Query memory and generate answer
        logger.info("\nStep 3: Query memory and generate answer")
        
        question = sample.get("input", "")
        gold_answers = sample.get("answers", [])
        
        logger.info(f"Question: {question}")
        logger.info(f"Standard answers: {gold_answers}")
        
        result = {
            "_id": sample.get("_id", sample_id),
            "sample_id": sample_id,
            "index": sample.get("index", sample_index),
            "question": question,
            "answers": gold_answers,
            "gold_answers": gold_answers,
            "num_chunks": len(context_chunks)
        }

        try:
            # Query memory to get research summary
            logger.info("Querying memory...")
            research_summary = agent.search_memory(question)
            logger.info("[OK] Memory query completed")
            logger.info(f"Research summary: {research_summary}")
            
            result["research_summary"] = research_summary
            
            # Generate final answer using working model API
            logger.info("Generating final answer...")
            prompt = make_prompt(research_summary, question)
            
            # Call working model API (use module-level import)
            working_client = OpenAI(api_key=working_api_key, base_url=working_base_url)
            response = working_client.chat.completions.create(
                model=working_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256
            )
            answer_text = response.choices[0].message.content.strip()
            
            logger.info(f"Model response: {answer_text}")
            
            pred_answer = answer_text
            result["response"] = answer_text
            result["pred"] = pred_answer
            
            # Calculate F1 score
            f1 = _calculate_f1(pred_answer, gold_answers) if pred_answer else 0.0
            result["f1"] = f1
            
            logger.info(f"Predicted answer: {pred_answer}")
            logger.info(f"Standard answers: {gold_answers}")
            logger.info(f"F1 score: {f1:.4f}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to process question: {e}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)
        
        # Save result
        results_file = os.path.join(sample_results_dir, "qa_result.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"\n[OK] Result saved to: {results_file}")
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("Processing completion statistics")
        logger.info(f"{'='*60}")
        logger.info(f"Sample ID: {sample_id}")
        logger.info(f"Number of context chunks: {len(context_chunks)}")
        logger.info(f"Predicted answer: {result.get('pred', 'N/A')}")
        logger.info(f"Standard answers: {gold_answers}")
        logger.info(f"F1 score: {result.get('f1', 0.0):.4f}")
        logger.info(f"Result saved to: {sample_results_dir}")
        
        return result
        
    except Exception as e:
        error_msg = f"Error processing sample {sample_index}: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "sample_id": sample.get("_id", f"sample-{sample_index}"),
            "error": error_msg
        }
    
    finally:
        # CRITICAL: Always cleanup GPU memory regardless of success or failure
        # This prevents GPU memory leaks between samples
        if agent is not None:
            try:
                logger.info("[CLEANUP] Starting GPU memory cleanup...")
                
                # Clear internal references to help garbage collection
                if hasattr(agent, '_memory_handler'):
                    handler = agent._memory_handler
                    logger.info(f"[CLEANUP] Found _memory_handler: {type(handler)}")
                    
                    # 1. Clear active agent's merged_cache AND model reference
                    if hasattr(handler, 'add_handler') and handler.add_handler.active_memory_agent:
                        active_agent = handler.add_handler.active_memory_agent
                        logger.info(f"[CLEANUP] Found active_memory_agent, has merged_cache: {hasattr(active_agent, 'merged_cache') and active_agent.merged_cache is not None}")
                        if hasattr(active_agent, 'merged_cache') and active_agent.merged_cache is not None:
                            del active_agent.merged_cache
                            active_agent.merged_cache = None
                        # CRITICAL: Clear model reference from MemoryAgent
                        if hasattr(active_agent, 'model'):
                            logger.info("[CLEANUP] Clearing active_agent.model")
                            active_agent.model = None
                        if hasattr(active_agent, 'tokenizer'):
                            active_agent.tokenizer = None
                    else:
                        logger.info(f"[CLEANUP] No active_memory_agent found, has_add_handler={hasattr(handler, 'add_handler')}")
                    
                    # 2. Clear inactive agents' references AND model references
                    if hasattr(handler, 'inactive_memory_agents'):
                        for inactive_agent in handler.inactive_memory_agents:
                            if hasattr(inactive_agent, 'merged_cache') and inactive_agent.merged_cache is not None:
                                del inactive_agent.merged_cache
                                inactive_agent.merged_cache = None
                            if hasattr(inactive_agent, '_cpu_cache'):
                                inactive_agent._cpu_cache = None
                            # CRITICAL: Clear model reference from each MemoryAgent
                            if hasattr(inactive_agent, 'model'):
                                inactive_agent.model = None
                            if hasattr(inactive_agent, 'tokenizer'):
                                inactive_agent.tokenizer = None
                        handler.inactive_memory_agents.clear()
                    
                    # 3. Clear HybridRouter's embedding model and cached embeddings FIRST
                    if hasattr(handler, 'query_handler') and hasattr(handler.query_handler, 'router'):
                        router = handler.query_handler.router
                        # Clear agent references in router FIRST
                        if hasattr(router, 'agent'):
                            for router_agent in router.agent:
                                if hasattr(router_agent, 'model'):
                                    router_agent.model = None
                                if hasattr(router_agent, 'tokenizer'):
                                    router_agent.tokenizer = None
                            router.agent.clear()
                        # Clear embedding model (SentenceTransformer uses GPU)
                        if hasattr(router, '_embedding_model') and router._embedding_model is not None:
                            if hasattr(router._embedding_model, 'model'):
                                # Move to CPU first to free GPU memory
                                try:
                                    router._embedding_model.model.cpu()
                                except Exception:
                                    pass
                                del router._embedding_model.model
                            del router._embedding_model
                            router._embedding_model = None
                        # Clear cached embeddings
                        if hasattr(router, '_summary_embeddings'):
                            router._summary_embeddings = None
                        if hasattr(router, '_text_chunk_embeddings'):
                            router._text_chunk_embeddings = None
                        if hasattr(router, '_text_chunks_per_block'):
                            router._text_chunks_per_block = []
                        if hasattr(router, '_chunk_to_block_map'):
                            router._chunk_to_block_map = []
                        if hasattr(router, '_bm25_scorer'):
                            router._bm25_scorer = None
                    
                    # 4. FINALLY: Clear the shared LLM model from AddHandler
                    # Must be done AFTER all MemoryAgents have their model references cleared
                    if hasattr(handler, 'add_handler'):
                        add_handler = handler.add_handler
                        logger.info(f"[CLEANUP] Found add_handler, has _shared_model: {hasattr(add_handler, '_shared_model') and add_handler._shared_model is not None}")
                        if hasattr(add_handler, 'active_memory_agent') and add_handler.active_memory_agent:
                            if hasattr(add_handler.active_memory_agent, 'model'):
                                add_handler.active_memory_agent.model = None
                            add_handler.active_memory_agent = None
                        if hasattr(add_handler, '_shared_model') and add_handler._shared_model is not None:
                            # CRITICAL: Move model to CPU first to free GPU memory
                            # This is more effective than just del
                            logger.info("[CLEANUP] Moving _shared_model to CPU and deleting...")
                            try:
                                add_handler._shared_model.cpu()
                                logger.info("[CLEANUP] _shared_model moved to CPU")
                            except Exception as e:
                                logger.warning(f"[CLEANUP] Failed to move model to CPU: {e}")
                            del add_handler._shared_model
                            add_handler._shared_model = None
                            logger.info("[CLEANUP] _shared_model deleted")
                        if hasattr(add_handler, '_shared_tokenizer'):
                            add_handler._shared_tokenizer = None
                        if hasattr(add_handler, '_shared_layer_devices'):
                            add_handler._shared_layer_devices = None
                    else:
                        logger.warning("[CLEANUP] No add_handler found in handler!")
                            
            except Exception as cleanup_error:
                logger.warning(f"Error during internal cleanup: {cleanup_error}")
            
            del agent
            logger.info("[CLEANUP] agent deleted")
        
        # Log memory before gc
        if torch.cuda.is_available() and logger:
            allocated_before_gc = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"[CLEANUP] GPU before gc.collect: allocated={allocated_before_gc:.2f}GB")
        
        # Force comprehensive GPU memory cleanup
        force_cleanup_gpu_memory()
        
        # Log GPU memory status for debugging
        if torch.cuda.is_available() and logger:
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU memory after cleanup: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
            except Exception:
                pass
        
        logger.info("Agent and GPU memory cleaned up")


# ========== Main Function ==========

def main():
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="KV-Cached Memory Agent + HotpotQA Evaluation")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to config file")
    parser.add_argument("--start-idx", type=int, default=0, help="Start sample index")
    parser.add_argument("--end-idx", type=int, default=None, help="End sample index (exclusive)")
    parser.add_argument("--ratio", type=float, default=None, help="Override evaluation ratio")
    
    # Working Generator configuration (for final answer generation)
    parser.add_argument("--working-api-key", type=str, default=None, help="Working model API Key")
    parser.add_argument("--working-base-url", type=str, default=None, help="Working model Base URL")
    parser.add_argument("--working-model", type=str, default=None, help="Working model name")
    
    args = parser.parse_args()
    
    # Load config
    config_path = resolve_eval_config_path(__file__, args.config)
    config = load_raw_config(config_path)
    if args.ratio is not None:
        config.setdefault("hotpotqa_eval", {})["ratio"] = args.ratio
    app_config = ensure_app_config(config)

    # Override working model config if provided
    question_answer_config = app_config.model.get_question_answer_openai_config().model_dump()
    working_api_key = args.working_api_key or question_answer_config['api_key']
    working_base_url = args.working_base_url or question_answer_config['base_url']
    working_model = args.working_model or question_answer_config['model']
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"hotpotqa_{timestamp}"
    os.environ['EVAL_SESSION_ID'] = session_id
    
    log_dir = app_config.logging.log_dir
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(PROJECT_ROOT, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"hotpotqa_eval_{timestamp}.log")
    logger = setup_logger(log_file)
    
    logger.info(f"Evaluation session ID: {session_id}")
    logger.info("=" * 60)
    logger.info("KV-Cached Memory Agent + HotpotQA Evaluation")
    logger.info("=" * 60)
    
    # Load data
    dataset_path = app_config.hotpotqa_eval.dataset_path
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(PROJECT_ROOT, dataset_path)
    
    logger.info(f"Dataset: {dataset_path}")
    all_samples = load_hotpotqa(dataset_path)
    logger.info(f"Total loaded {len(all_samples)} samples")
    
    # Set end index
    if args.end_idx is None:
        args.end_idx = len(all_samples)
    
    # Apply ratio
    ratio = app_config.hotpotqa_eval.ratio
    if ratio < 1.0:
        args.end_idx = min(args.end_idx, max(1, int(len(all_samples) * ratio)))
    
    logger.info(f"Processing range: {args.start_idx} to {args.end_idx-1} (total {args.end_idx - args.start_idx} samples)")
    
    # Validate range
    if args.start_idx < 0 or args.start_idx >= len(all_samples):
        logger.error(f"Error: Start index {args.start_idx} out of range")
        return
    
    if args.end_idx > len(all_samples):
        logger.warning(f"Warning: End index {args.end_idx} out of range, adjusted to {len(all_samples)}")
        args.end_idx = len(all_samples)
    
    if args.start_idx >= args.end_idx:
        logger.error("Error: Start index must be less than end index")
        return
    
    # Process samples
    outdir = app_config.hotpotqa_eval.output_dir
    if not os.path.isabs(outdir):
        outdir = os.path.join(PROJECT_ROOT, outdir)
    
    max_tokens = app_config.hotpotqa_eval.max_tokens_per_chunk
    
    sample_indices = list(range(args.start_idx, args.end_idx))
    logger.info("Starting serial processing of samples...")
    
    all_results = []
    for sample_idx in tqdm(sample_indices, desc="Processing samples"):
        sample = all_samples[sample_idx]
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting to process sample {sample_idx}/{len(all_samples)-1}")
        logger.info(f"{'='*80}")
        
        # Log GPU memory BEFORE processing
        if torch.cuda.is_available():
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            reserved_before = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"[GPU BEFORE] allocated={allocated_before:.2f}GB, reserved={reserved_before:.2f}GB")
        
        try:
            result = process_sample(
                sample, 
                sample_idx, 
                outdir,
                working_api_key,
                working_base_url,
                working_model,
                app_config,
                max_tokens=max_tokens,
                logger=logger
            )
            logger.info(f"[OK] Sample {sample_idx} processing completed")
            all_results.append(result)
        except Exception as e:
            logger.error(f"[ERROR] Sample {sample_idx} processing failed: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "sample_id": sample.get("_id", f"sample-{sample_idx}"),
                "error": str(e)
            })
        
        # Log GPU memory AFTER processing (should be similar to BEFORE if cleanup is thorough)
        if torch.cuda.is_available():
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            reserved_after = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"[GPU AFTER] allocated={allocated_after:.2f}GB, reserved={reserved_after:.2f}GB")
            # Warn if significant memory increase detected
            if allocated_after - allocated_before > 0.5:  # More than 0.5GB increase
                logger.warning(f"[WARNING] GPU memory increased by {allocated_after - allocated_before:.2f}GB after sample {sample_idx}")
    
    # Calculate statistics
    f1_scores = [r["f1"] for r in all_results if "f1" in r]
    runtime_models = app_config.get_runtime_model_summary()
    
    if all_results:
        os.makedirs(outdir, exist_ok=True)
        batch_results_payload = {
            "model": runtime_models["memory_agent_model"],
            "models": runtime_models,
            "dataset": dataset_path,
            "start_idx": args.start_idx,
            "end_idx": args.end_idx - 1,
            "individual_results": all_results,
        }
        summary_file = os.path.join(outdir, f"batch_results_{args.start_idx}_{args.end_idx-1}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results_payload, f, ensure_ascii=False, indent=2)
        logger.info(f"[OK] Batch results summary saved: {summary_file}")
        
        if f1_scores:
            avg_f1 = sum(f1_scores) / len(f1_scores)
            total_samples = args.end_idx - args.start_idx
            success_count = len(f1_scores)
            
            statistics = {
                "model": runtime_models["memory_agent_model"],
                "models": runtime_models,
                "dataset": dataset_path,
                "total_samples": total_samples,
                "success_count": success_count,
                "failed_count": total_samples - success_count,
                "success_rate": success_count / total_samples if total_samples > 0 else 0.0,
                "avg_f1": avg_f1,
                "f1_scores": f1_scores,
                "start_idx": args.start_idx,
                "end_idx": args.end_idx - 1
            }
            
            stats_file = os.path.join(outdir, f"batch_statistics_{args.start_idx}_{args.end_idx-1}.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, ensure_ascii=False, indent=2)
            logger.info(f"[OK] Batch test statistics saved: {stats_file}")
            
            logger.info(f"\n{'='*60}")
            logger.info("Batch Test Statistics")
            logger.info(f"{'='*60}")
            logger.info(f"Processed samples: {total_samples}")
            logger.info(f"Successfully answered questions: {success_count}")
            logger.info(f"Failed questions: {total_samples - success_count}")
            logger.info(f"Success rate: {statistics['success_rate']:.2%}")
            logger.info(f"Average F1 score: {avg_f1:.4f}")
            logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
