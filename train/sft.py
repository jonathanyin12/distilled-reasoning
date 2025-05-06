import os
from dataclasses import dataclass, field
from typing import Optional

import transformers
import trl
from datasets import load_dataset


@dataclass
class TrainingConfig:
    base_model: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    dataset_file_path: str = field(
        default="jonathanyin/aime_1983_2023_deepseek-r1_traces_16384"
    )
    wandb_project: Optional[str] = field(default="LLM Reasoning")
    wandb_entity: Optional[str] = field(default="jonathanyin-yale")

    def __post_init__(self):
        os.environ["WANDB_PROJECT"] = self.wandb_project
        os.environ["WANDB_ENTITY"] = self.wandb_entity


def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    training_config, training_args = parser.parse_args_into_dataclasses()

    print(f"Base model: {training_config.base_model}")
    print(f"Dataset: {training_config.dataset_file_path}")
    print(f"Training args: {training_args}")

    # loading model
    kwargs = {}
    if "70B" in training_config.base_model:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "attn_implementation": "flash_attention_2",
            "use_cache": False,
        }
        model = transformers.AutoModelForCausalLM.from_pretrained(
            training_config.base_model, **kwargs
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            training_config.base_model
        )

    dataset = load_dataset(training_config.dataset_file_path)

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        training_config.base_model, use_fast=True
    )
    if "Llama" in training_config.base_model:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in training_config.base_model:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args.use_liger_kernel = True

    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if "test" in dataset else dataset["train"],
        args=training_args,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
