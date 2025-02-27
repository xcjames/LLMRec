import argparse
import os

import sys
from typing import List

import torch
import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from utils import *
from collator import Collator

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

def train(args):

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1  # Distributed Data Parallel
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    # Model parallelism setup: Specify the device placement for different layers
    device_map = {
        "transformer.h.0": 0,  # First layer on GPU 0
        "transformer.h.1": 1,  # Second layer on GPU 1
        "transformer.h.2": 2,  # Third layer on GPU 2
        # You can continue assigning other layers to GPUs based on your setup
    }

    config = LlamaConfig.from_pretrained(args.base_model)
    tokenizer = LlamaTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.model_max_length,
        padding_side="right",
    )
    tokenizer.pad_token_id = 0
    gradient_checkpointing = True

    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    collator = Collator(args, tokenizer)

    # Load the model for model parallelism
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        low_cpu_mem_usage=True,
        device_map=device_map,  # Apply the device map
    )
    model.resize_token_embeddings(len(tokenizer))

    # Set model parallelism flags
    model.is_parallelizable = True
    model.model_parallel = True
    model.config.use_cache = False  # Ensure caching is disabled during training

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
            gradient_checkpointing=gradient_checkpointing,
            evaluation_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=5,
            load_best_model_at_end=True,
            deepspeed=args.deepspeed,
            ddp_find_unused_parameters=False if ddp else None,
            report_to=None,
            eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()

    train(args)
