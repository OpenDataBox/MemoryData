"""Evaluation script for LoComo dataset."""
import argparse
import gc
import json
import logging
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from openai import OpenAI

from config import ensure_app_config, load_raw_config
from evaluation.config_utils import resolve_eval_config_path
from evaluation.locomo.load_dataset import (
    filter_dataset_by_questions,
    load_locomo_dataset,
    load_specific_questions,
)
from evaluation.locomo.utils import (
    aggregate_metrics,
    calculate_metrics,
    extract_answer_from_xml,
)
from src.conversation_manager.factory import create_chat_manager

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = "evaluation/locomo/config.yaml"


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


def setup_logger(log_file: str) -> logging.Logger:
    """Setup logging configuration."""
    # Configure root logger to capture all logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
    
    # Return specific logger for eval
    logger = logging.getLogger('locomo_eval')
    return logger


def evaluate_dataset(config: dict, logger: logging.Logger):
    """Evaluate on LoComo dataset."""
    app_config = ensure_app_config(config)

    # Generate unique session ID for this evaluation run
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = app_config.get_model_label()
    session_id = f"locomo_{os.path.basename(model_id)}_{timestamp}"
    os.environ['EVAL_SESSION_ID'] = session_id
    logger.info(f"Evaluation session ID: {session_id}")
    logger.info("Note: Using session-specific metadata to support concurrent evaluations")
    
    # Load dataset
    dataset_path = app_config.locomo_eval.dataset_path
    if not os.path.isabs(dataset_path):
        # Path is relative to project root, not to this file
        project_root = Path(__file__).parent.parent.parent
        dataset_path = os.path.join(project_root, dataset_path)
    
    logger.info(f"Loading dataset from {dataset_path}")
    samples = load_locomo_dataset(dataset_path)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Check if specific questions should be used
    specific_questions_path = app_config.locomo_eval.specific_questions_path
    if specific_questions_path:
        if not os.path.isabs(specific_questions_path):
            project_root = Path(__file__).parent.parent.parent
            specific_questions_path = os.path.join(project_root, specific_questions_path)
        
        logger.info(f"Loading specific questions from {specific_questions_path}")
        specific_questions = load_specific_questions(specific_questions_path)
        logger.info(f"Loaded {len(specific_questions)} specific questions")
        
        # Filter dataset to only include specific questions
        samples = filter_dataset_by_questions(samples, specific_questions)
        logger.info(f"Filtered to {len(samples)} samples with specific questions")
    
    # Apply ratio
    ratio = app_config.locomo_eval.ratio
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")
    
    # Results storage
    results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)
    
    # Evaluate each sample
    for sample_idx, sample in enumerate(samples):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing sample {sample_idx + 1}/{len(samples)}")
        logger.info(f"{'='*80}")
        
        # Log GPU memory BEFORE processing
        allocated_before = None
        if torch.cuda.is_available():
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            reserved_before = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"[GPU BEFORE] allocated={allocated_before:.2f}GB, reserved={reserved_before:.2f}GB")
        
        agent = None  # Initialize agent to None for proper cleanup in finally block
        
        try:
            # Create agent
            agent = create_chat_manager(**app_config.to_chat_manager_kwargs())

            # Set up working client for final answer generation
            openai_config = app_config.model.get_question_answer_openai_config().model_dump()
            working_client = OpenAI(
                api_key=openai_config['api_key'],
                base_url=openai_config['base_url']
            )
            working_model = openai_config['model']
            
            # Store conversations
            logger.info("\n--- Storing Conversation Memories ---")
            conversation_auto_save = app_config.locomo_eval.conversation_auto_save
            
            for session_key, session in sample.conversation.sessions.items():
                for turn in session.turns:
                    if conversation_auto_save:
                        memory_text = f"[{session.date_time}] {turn.speaker}: {turn.text}\n"
                    else:
                        memory_text = f"[{session.date_time}] {turn.speaker}: {turn.text}"
                    
                    try:
                        # Use direct memory addition instead of chat
                        agent.add_memory(memory_text)
                        logger.info(f"Stored: {memory_text[:100]}...")
                    except Exception as e:
                        logger.error(f"Error storing memory: {e}")
            
            # Answer questions
            logger.info("\n--- Answering Questions ---")
            allowed_categories = app_config.locomo_eval.categories
            
            for qa in sample.qa:
                if qa.category not in allowed_categories:
                    continue
                
                total_questions += 1
                category_counts[qa.category] += 1
                
                # Determine reference answer based on category
                if qa.category == 5:
                    # Adversarial question
                    # If both answer and adversarial_answer exist, answer is correct, adversarial_answer is wrong
                    # Otherwise, 'Not mentioned' is correct, adversarial_answer is wrong
                    if qa.answer and qa.adversarial_answer:
                        reference_answer = qa.answer
                        wrong_answer = qa.adversarial_answer
                    else:
                        reference_answer = "Not mentioned in the conversation"
                        wrong_answer = qa.adversarial_answer or "Unknown"
                else:
                    # Other categories use answer field
                    reference_answer = qa.answer
                
                # Skip if no reference answer available
                if not reference_answer:
                    logger.warning(f"Skipping question without reference answer: {qa.question}")
                    continue
                
                # Build specialized instructions based on category
                if qa.category == 5:
                    # Adversarial question - randomize answer order
                    answer_tmp = []
                    if random.random() < 0.5:
                        answer_tmp.append(reference_answer)
                        answer_tmp.append(wrong_answer)
                    else:
                        answer_tmp.append(wrong_answer)
                        answer_tmp.append(reference_answer)
                    
                    instructions = f"""### CRITICAL: THE "REJECTION FIRST" PROTOCOL
Your goal is to PROVE that the answer is MISSING. You must actively try to disqualify any potential evidence.

1. **Default Verdict:** Start with the assumption that the correct answer is "Not mentioned".
2. **The "Exact Match" Test:**
   - Does the text contain the **exact keywords** or **synonyms** for the specific detail asked?
   - If NO -> Select "Not mentioned".
   - If YES, but it refers to a different context/person/time -> Select "Not mentioned".

3. **The "No Inference" Firewall:**
   - **Hypothetical Scenario:** If the text says "She likes fruit" and the question asks "Does she like apples?", you MUST select "Not mentioned" (because she might only like bananas).
   - **Probability vs. Certainty:** Even if an answer is 99% likely to be true based on context, if it is not 100% stated -> Select "Not mentioned".

4. **Final Check:** Before selecting a specific option, ask yourself: "Can I underline the exact sentence that proves this?" If you cannot physically point to the sentence -> Select "**Not mentioned** or **no reference**".

YOU MUST Select the correct answer: '{answer_tmp[0]}' or '{answer_tmp[1]}'. Provide ONLY the selected answer without explanation."""
                elif qa.category == 2:  # Date/Time Questions
                    instructions = """You are a precise date extraction and normalization assistant.
Your task is to answer the question based on the document and STRICTLY adhere to the formatting rules below.

### STEP 1: DETERMINE OUTPUT MODE
Analyze the `Question` text provided at the bottom.
1.  **IF the question starts with the word "When" (case-insensitive):**
    * You MUST use **MODE A (Calendar Date)**.
2.  **OTHERWISE (for all other questions like "How long", "What duration", etc.):**
    * You MUST use **MODE B (Duration & Time Ago)**.

### STEP 2: APPLY FORMATTING RULES

**MODE A: Calendar Date (Standardized)**
* **Goal:** Extract a specific point in time.
* **Format Rules:** Convert to `[Day] [Full Month Name] [Year]`.
* **Allowed Templates:**
    * Full Date: `[Day] [Full Month Name] [Year]` (e.g., "6 July 2023")
    * Month/Year: `[Full Month Name] [Year]` (e.g., "June 2023")
    * Year Only: `[Year]` (e.g., "2023")

**MODE B: Duration & Time Ago**
* **Goal:** Extract a time span or relative past interval.
* **Allowed Templates:**
    * Standard Duration: `[Number] [Unit]`      (e.g., "3 years", "6 months")
    * Past Reference:    `[Number] [Unit] ago`  (e.g., "10 years ago", "2 weeks ago")

### STEP 3: CRITICAL CONSTRAINTS
1.  **ANTI-REFUSAL:** **NEVER** output phrases like "not mentioned", "unknown", "N/A", or "does not say".
    * If the exact answer is missing, you **MUST infer** the most reasonable date/duration from the context and format it.
2.  **CLEAN OUTPUT:** Output **ONLY** the final string. No intro text, no punctuation (periods) at the end."""
                elif qa.category == 1:  # Fact Retrieval/General
                    instructions = """
                 Based on the text below, analyze the context to provide the best answer to the question.

### PROCESS
1. **Understand:** Fully grasp the specific details requested by the question.
2. **Reason:** Analyze the text in relation to the question to infer or locate the correct answer.
3. **Extract:** Formulate the final answer using the **exact wording and terminology** directly from the context whenever possible.
4.   For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" .
### CRITICAL CONSTRAINTS
* **Strict Relevance:** **NEVER** output content unrelated to the question. Focus ONLY on the specific details requested.
* **Anti-Refusal:** **NEVER** say "not mentioned". Even if the answer is not explicitly stated, you **MUST** infer the **most probable answer** based on the provided evidence.
* **Format:** The final OUTPUT must be a **short phrase** (under 10 words). NO sentences.
"""
                elif qa.category == 3:  # Analysis/Inference Questions
                    instructions = """Based on the text below, write an answer in the form of **a short phrase** for the following question, not a sentence.
### CRITICAL: CONDITIONAL FORMATTING
1. **"Would" Questions:**
   * **Case A (Selection):** If the question asks to choose between options (often containing "or", e.g., "Would X or Y happen?"), output the **specific option** selected.
     * *Example:** "Would he use Python or Java?" -> "Python"
   * **Case B (Prediction/Judgment):** If the question asks for a yes/no prediction or feasibility, your answer **MUST** start with **"Likely yes"** or **"Likely no"**.
     * *Example:* "Would this be accepted?" -> "Likely yes, due to..."
2. **Other Questions:**
   * Output a **short phrase**.
   * Do NOT use full sentences.

### CRITICAL: ANTI-REFUSAL POLICY
* **NEVER** output refusal phrases.
* **ALWAYS** infer the best possible answer based on available evidence."""
                elif qa.category == 4:
                    # Detailed question
                    instructions = """
        You are an extractive QA assistant. Your goal is to extract the exact answer substring from the text.
INSTRUCTIONS:
1. Locate the exact answer in the TEXT.
2. Output **ONLY** the key entity, date, name, or short phrase representing the answer.
3. **DO NOT** use full sentences. **DO NOT** add filler words like "The answer is", "a", "the", "she is".
4. Keep it as short as possible (ideally less than 10 words).
5. Use the EXACT wording from the text if possible.
* **Format:** The final OUTPUT must be a **short phrase** (under 10 words).
        
"""
                else:
                    # Other categories
                    instructions = """Use DATE of CONVERSATION to answer with an approximate date. Write an answer in the form of a short phrase. Answer with exact words from the context whenever possible. Short answer:"""
                
                logger.info(f"\nQuestion {total_questions} (Category {qa.category}): {qa.question}")
                
                try:
                    # 1. Search memory for context
                    logger.info("Searching memory...")
                    research_summary = agent.search_memory(qa.question)
                    logger.info(f"Research summary: {research_summary[:200]}...")
                    
                    system_content = f"""You are an expert at answering questions based on conversation history.
                                            Instructions:{instructions}"""
                    user_content = f"""Question: {qa.question}

                                    Context:{research_summary}"""
                    # 3. Generate final answer using working model
                    logger.info("Generating final answer...")
                    response = working_client.chat.completions.create(
                        model=working_model,
                        # messages=[{"role": "user", "content": prompt}],
                        messages=[
                                    {"role": "system", "content": system_content},
                                    {"role": "user", "content": user_content}
                                ],
                        temperature=0,
                        max_tokens=256
                    )
                    prediction = response.choices[0].message.content.strip()
                    logger.info(f"Raw prediction: {prediction}")
                    
                    # Extract answer from XML tags if needed (though instructions were updated to be direct)
                    processed_prediction = extract_answer_from_xml(prediction, qa.category)
                    logger.info(f"Processed prediction: {processed_prediction}")
                    logger.info(f"Reference: {reference_answer}")
                    
                    queried_memory = research_summary
                except Exception as e:
                    logger.error(f"Error answering question: {e}")
                    prediction = "ERROR"
                    processed_prediction = "ERROR"
                    queried_memory = None
                
                # Calculate metrics using the processed prediction
                metrics = calculate_metrics(processed_prediction, reference_answer)
                all_metrics.append(metrics)
                all_categories.append(qa.category)
                
                results.append({
                    "sample_id": sample_idx,
                    "question": qa.question,
                    "prediction": prediction,  # Store raw prediction
                    "processed_prediction": processed_prediction,  # Store processed prediction
                    "reference": reference_answer,
                    "category": qa.category,
                    "metrics": metrics,
                    "queried_memory": queried_memory
                })
        
        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
        
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
                    # Warn if significant memory increase detected
                    if allocated_before is not None and allocated - allocated_before > 0.5:  # More than 0.5GB increase
                        logger.warning(f"[WARNING] GPU memory increased by {allocated - allocated_before:.2f}GB after sample {sample_idx}")
                except Exception:
                    pass
            
            logger.info(f"Agent and GPU memory cleaned up for sample {sample_idx}")
    
    # Aggregate metrics
    aggregate_results = aggregate_metrics(all_metrics, all_categories)
    runtime_models = app_config.get_runtime_model_summary()
    
    # Prepare final results
    final_results = {
        "model": runtime_models["memory_agent_model"],
        "models": runtime_models,
        "dataset": dataset_path,
        "total_questions": total_questions,
        "category_distribution": {str(cat): count for cat, count in category_counts.items()},
        "aggregate_metrics": aggregate_results,
        "individual_results": results
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = app_config.locomo_eval.output_dir
    if not os.path.isabs(output_dir):
        project_root = Path(__file__).parent.parent.parent
        output_dir = os.path.join(project_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"locomo_eval_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to {output_file}")
    
    # Log summary
    logger.info("\n" + "="*80)
    logger.info("Evaluation Summary")
    logger.info("="*80)
    logger.info(f"Total questions: {total_questions}")
    logger.info("\nCategory Distribution:")
    for category, count in sorted(category_counts.items()):
        logger.info(f"  Category {category}: {count} ({count/total_questions*100:.1f}%)")
    
    logger.info("\nOverall Metrics:")
    for metric, stats in aggregate_results['overall'].items():
        logger.info(f"  {metric}: mean={stats['mean']:.4f}, median={stats['median']:.4f}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate on LoComo dataset")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH,
                       help="Path to config file")
    parser.add_argument("--model_id", type=str, help="Override model ID")
    parser.add_argument("--dataset", type=str, help="Override dataset path")
    parser.add_argument("--ratio", type=float, help="Override evaluation ratio")
    parser.add_argument("--conversation_auto_save", action="store_true",
                       help="Enable auto_save for conversation turns")
    args = parser.parse_args()
    
    # Load config
    config_path = resolve_eval_config_path(__file__, args.config)
    config = load_raw_config(config_path)

    # Override config with command line args
    if args.model_id:
        config.setdefault('tokenizer', {})['model_id'] = args.model_id
        config.setdefault('model', {}).setdefault('memory_agent_model', {})['model_id'] = args.model_id
    if args.dataset:
        config['locomo_eval']['dataset_path'] = args.dataset
    if args.ratio:
        config['locomo_eval']['ratio'] = args.ratio
    if args.conversation_auto_save:
        config['locomo_eval']['conversation_auto_save'] = True

    app_config = ensure_app_config(config)
    
    # Ensure directories exist (relative to project root)
    os.makedirs(PROJECT_ROOT / 'evaluation' / 'locomo' / 'logs', exist_ok=True)
    os.makedirs(PROJECT_ROOT / 'evaluation' / 'locomo' / 'results', exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = app_config.logging.log_dir
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(PROJECT_ROOT, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"eval_{timestamp}.log")
    logger = setup_logger(log_file)
    
    logger.info("Starting evaluation")
    logger.info(f"Config: {app_config.model_dump()}")

    # Run evaluation
    evaluate_dataset(app_config, logger)
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
