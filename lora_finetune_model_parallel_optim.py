###################################
import argparse
import os
import sys
from typing import List

import torch
import transformers
from accelerate import dispatch_model  # Model Parallel Dispatching
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig

from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from utils import *
from collator import Collator


def train(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    world_size = torch.cuda.device_count()  # Get number of GPUs
    use_model_parallel = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # ✅ 1️⃣ Optimized Model Parallelism
    device_map = "sequential" if use_model_parallel else None  # Force logical layer distribution
    offload_folder = args.output_dir + "/offload" if use_model_parallel else None  # CPU Offloading

    if local_rank == 0:
        print(vars(args))

    # Load tokenizer and model config
    config = LlamaConfig.from_pretrained(args.base_model)
    tokenizer = LlamaTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.model_max_length,
        padding_side="right",
    )
    tokenizer.pad_token_id = 0

    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)

    if local_rank == 0:
        print(f"Added {add_num} new tokens.")
        print(f"Training data size: {len(train_data)}")
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    collator = Collator(args, tokenizer)

    # ✅ 2️⃣ Load Model with Offloading & Model Parallelism
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,  # Lower Precision for Memory Efficiency
        device_map=device_map,  # Distribute Layers Across GPUs
        offload_folder=offload_folder,  # Offload Some Layers to Disk if Needed
    )
    model.resize_token_embeddings(len(tokenizer))

    # ✅ 3️⃣ Apply LoRA Configurations
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        modules_to_save=args.lora_modules_to_save.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # ✅ 4️⃣ Resume from Checkpoint if Available
    if args.resume_from_checkpoint:
        checkpoint_name = os.path.join(args.resume_from_checkpoint, "adapter_model.bin")
        args.resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            if local_rank == 0:
                print(f"Resuming from checkpoint: {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            if local_rank == 0:
                print(f"Checkpoint not found: {checkpoint_name}")

    # ✅ 5️⃣ Freeze LoRA-unrelated Parameters
    for name, param in model.named_parameters():
        if "original_module" in name and any(module in name for module in lora_config.modules_to_save):
            param.requires_grad = False

    if local_rank == 0:
        model.print_trainable_parameters()

    # ✅ 6️⃣ Dispatch Model with Parallelism + Offloading
    model = dispatch_model(model)

    # ✅ 7️⃣ Training Arguments with Optimizations
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            fp16=args.fp16,
            bf16=args.bf16,
            logging_steps=args.logging_step,
            optim=args.optim,
            gradient_checkpointing=True,
            evaluation_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=5,
            load_best_model_at_end=True,
            deepspeed=args.deepspeed,
            report_to=None,
            eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
    )

    model.config.use_cache = False

    # ✅ 8️⃣ Enable `torch.compile()` for Even Faster Performance
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model, mode="reduce-overhead")  # Optimized Compilation

    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized LoRA Model Parallel Fine-Tuning")
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()

    train(args)
