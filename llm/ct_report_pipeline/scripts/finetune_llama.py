"""
Fine-tune Llama-3.2-3B-Instruct for CT Report Generation

Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning.
Requires: transformers, peft, bitsandbytes, accelerate, datasets

Usage:
    python scripts/finetune_llama.py --epochs 3 --batch_size 1
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import Dataset


def load_training_data(jsonl_path: str) -> Dataset:
    """Load training data from JSONL file."""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # Support new format with messages
            if 'messages' in item:
                messages = item['messages']
                user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
                asst_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), '')
                data.append({
                    'prompt': user_msg,
                    'response': asst_msg,
                })
            # Support old format with prompt/response
            elif 'prompt' in item and 'response' in item:
                data.append({
                    'prompt': item['prompt'],
                    'response': item['response'],
                })
    
    print(f"Loaded {len(data)} training samples")
    return Dataset.from_list(data)


def format_training_example(example: dict, tokenizer) -> dict:
    """Format example for training with chat template."""
    
    # System prompt matching the one in report_generator
    system_prompt = """You are an experienced radiologist assistant. Generate professional CT chest reports based on provided nodule measurements.

Rules:
1. Use ONLY the provided measurements - do not fabricate data
2. Follow the standard radiology report structure
3. Output in English only
4. Include Lung-RADS 2022 category assessment
5. Leave uncertain fields empty or state "Not evaluated"
6. Be concise and clinically relevant"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example['prompt']},
        {"role": "assistant", "content": example['response']},
    ]
    
    # Apply chat template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        # Fallback format
        text = f"""### System:
{system_prompt}

### User:
{example['prompt']}

### Assistant:
{example['response']}"""
    
    return {'text': text}


def tokenize_function(examples, tokenizer, max_length=2048):
    """Tokenize examples for training."""
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors=None,
    )


def load_config():
    """Load settings from pipeline_config.yaml."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config.config_loader import load_config as load_pipeline_config

    config_path = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"
    return load_pipeline_config(str(config_path))


def main():
    # Load config first
    config = load_config()
    llm_config = config.get('llm', {})
    
    parser = argparse.ArgumentParser(description="Fine-tune Llama for CT reports")
    parser.add_argument("--model_name", type=str, 
                        default=llm_config.get('model_name', 'meta-llama/Llama-3.2-1B-Instruct'))
    parser.add_argument("--data_path", type=str, default="assets/data/finetune_train.jsonl")
    parser.add_argument("--val_data_path", type=str, default="assets/data/finetune_val.jsonl")
    parser.add_argument("--output_dir", type=str, default="assets/models/lora_ct_report")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_length", type=int, 
                        default=llm_config.get('max_length', 2048))
    parser.add_argument("--use_8bit", action="store_true", 
                        default=llm_config.get('load_in_8bit', False),
                        help="Use 8-bit quantization")
    
    args = parser.parse_args()
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"lora_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CT Report LLM Fine-Tuning (LoRA)")
    print("=" * 60)
    print(f"Config: {Path(__file__).parent.parent / 'config' / 'pipeline_config.yaml'}")
    print(f"Model: {args.model_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"LoRA rank: {args.lora_r}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("\n[2/5] Loading model...")
    if args.use_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # Enable gradient checkpointing properly
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # Enable input embeddings gradients for gradient checkpointing compatibility
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # Configure LoRA
    print("\n[3/5] Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare data
    print("\n[4/5] Preparing training data...")
    dataset = load_training_data(args.data_path)
    
    # Format examples
    dataset = dataset.map(
        lambda x: format_training_example(x, tokenizer),
        remove_columns=dataset.column_names,
    )
    
    # Tokenize
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=['text'],
    )
    
    # Add labels
    dataset = dataset.map(
        lambda x: {'labels': x['input_ids'].copy()},
    )
    
    print(f"Training samples: {len(dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        optim="adamw_torch",
    )
    
    # Load validation data
    val_dataset = None
    if Path(args.val_data_path).exists():
        print(f"\n[4b/5] Loading validation data...")
        val_dataset = load_training_data(args.val_data_path)
        val_dataset = val_dataset.map(
            lambda x: format_training_example(x, tokenizer),
            remove_columns=val_dataset.column_names,
        )
        val_dataset = val_dataset.map(
            lambda x: tokenize_function(x, tokenizer, args.max_length),
            batched=True,
            remove_columns=['text'],
        )
        val_dataset = val_dataset.map(
            lambda x: {'labels': x['input_ids'].copy()},
        )
        print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n[5/5] Starting training...")
    print("-" * 60)
    
    trainer.train()
    
    # Evaluate on validation set
    eval_results = {}
    if val_dataset:
        print("\n" + "=" * 60)
        print("Evaluation Results (Loss)")
        print("=" * 60)
        eval_results = trainer.evaluate()
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
    
    # Generation-based evaluation (BLEU, METEOR)
    print("\n" + "=" * 60)
    print("Generation Evaluation (BLEU, METEOR)")
    print("=" * 60)
    gen_metrics = evaluate_generation(
        model, tokenizer, args.val_data_path, 
        max_samples=5, max_new_tokens=args.max_length // 2
    )
    for key, value in gen_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    eval_results.update(gen_metrics)
    
    # Save final model
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    final_path = output_dir / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"\nLoRA weights saved to: {final_path}")
    
    # Save training metrics
    metrics_file = output_dir / "training_metrics.json"
    metrics = {
        "model": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size * args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "train_samples": len(dataset),
        "val_samples": len(val_dataset) if val_dataset else 0,
        "eval_results": {k: float(v) if isinstance(v, (int, float)) else v for k, v in eval_results.items()},
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")
    
    print("\nTo use the fine-tuned model, update pipeline_config.yaml:")
    print(f"""
llm:
  model_name: "{args.model_name}"
  lora_path: "{final_path}"
""")


def evaluate_generation(model, tokenizer, val_data_path, max_samples=5, max_new_tokens=512):
    """Evaluate generation quality with BLEU and METEOR."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except ImportError:
        print("  NLTK not installed. Skipping BLEU/METEOR evaluation.")
        return {}
    
    # Load validation data (raw)
    val_data = []
    with open(val_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if 'messages' in item:
                messages = item['messages']
                user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
                asst_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), '')
                val_data.append({'prompt': user_msg, 'reference': asst_msg})
            elif 'prompt' in item and 'response' in item:
                val_data.append({'prompt': item['prompt'], 'reference': item['response']})
    
    if not val_data:
        return {}
    
    # Sample for evaluation
    import random
    samples = random.sample(val_data, min(max_samples, len(val_data)))
    
    bleu_scores = []
    meteor_scores = []
    format_compliance = []
    smoother = SmoothingFunction()
    
    model.eval()
    print(f"  Evaluating {len(samples)} samples...")
    
    for i, sample in enumerate(samples):
        # Generate
        messages = [
            {"role": "user", "content": sample['prompt']},
        ]
        
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = f"### User:\n{sample['prompt']}\n\n### Assistant:\n"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        reference = sample['reference']
        
        # Tokenize for BLEU
        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()
        
        # BLEU score
        try:
            bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoother.method1)
            bleu_scores.append(bleu)
        except:
            pass
        
        # METEOR score
        try:
            meteor = meteor_score([ref_tokens], gen_tokens)
            meteor_scores.append(meteor)
        except:
            pass
        
        # Format compliance check
        has_report_id = "Report ID:" in generated
        has_technique = "Technique:" in generated
        has_findings = "Findings:" in generated or "Lungs:" in generated
        has_lung_rads = "Lung-RADS" in generated or "Category:" in generated
        has_recommendation = "Recommendation:" in generated
        
        compliance = sum([has_report_id, has_technique, has_findings, has_lung_rads, has_recommendation]) / 5
        format_compliance.append(compliance)
    
    results = {}
    if bleu_scores:
        results['bleu'] = sum(bleu_scores) / len(bleu_scores)
    if meteor_scores:
        results['meteor'] = sum(meteor_scores) / len(meteor_scores)
    if format_compliance:
        results['format_compliance'] = sum(format_compliance) / len(format_compliance)
    
    return results


if __name__ == "__main__":
    main()



