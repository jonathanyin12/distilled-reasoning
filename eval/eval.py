import argparse
import os
from datetime import datetime

from aggregate_results import aggregate_results
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_input import GenerationParameters
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.utils import EnvConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--task", type=str, default="lighteval|aime24|0|0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--max_model_length", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--custom_tasks_directory", type=str, default=None)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--data_parallel_size", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    return parser.parse_args()


def main():
    script_start_time = datetime.now()
    args = parse_args()

    max_model_length = args.max_model_length
    if args.max_model_length is None:
        print("max_model_length not set. Setting it to max_new_tokens.")
        max_model_length = args.max_new_tokens
    elif args.max_model_length == -1:
        print("max_model_length is -1. Setting it to None.")
        max_model_length = None

    system_prompt = None
    if args.system_prompt is not None and os.path.exists(args.system_prompt):
        with open(args.system_prompt, "r") as f:
            system_prompt = f.read()

    output_dir = args.output_dir.replace("/", "_")
    for i in range(args.num_runs):
        run_start_time = datetime.now()
        current_seed = args.seed + i
        print(f"--- Starting run {i + 1}/{args.num_runs} with seed {current_seed} ---")

        env_config = EnvConfig()
        evaluation_tracker = EvaluationTracker(
            output_dir=output_dir,
            save_details=True,
            push_to_hub=False,
            push_to_tensorboard=False,
            public=False,
            hub_results_org=None,
        )

        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.VLLM,
            env_config=env_config,
            job_id=0,
            dataset_loading_processes=1,
            custom_tasks_directory=args.custom_tasks_directory,
            override_batch_size=-1,
            num_fewshot_seeds=1,
            max_samples=None,
            use_chat_template=args.use_chat_template,
            system_prompt=system_prompt,
            load_responses_from_details_date_id=None,
        )

        model_config = VLLMModelConfig(
            pretrained=args.model,
            dtype=args.dtype,
            seed=current_seed,
            use_chat_template=args.use_chat_template,
            max_model_length=max_model_length,
            gpu_memory_utilization=args.gpu_memory_utilization,
            data_parallel_size=args.data_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size,
            generation_parameters=GenerationParameters(
                max_new_tokens=args.max_new_tokens,
                seed=current_seed,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            ),
        )

        print(model_config)
        pipeline = Pipeline(
            tasks=args.task,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model_config=model_config,
            metric_options={},
        )

        pipeline.evaluate()
        pipeline.show_results()
        pipeline.save_and_push_results()

        run_end_time = datetime.now()
        print(
            f"--- Run {i + 1} took {(run_end_time - run_start_time).total_seconds():.2f} seconds ---"
        )

    script_end_time = datetime.now()
    print(
        f"=== All {args.num_runs} runs completed in {(script_end_time - script_start_time).total_seconds():.2f} seconds ==="
    )

    aggregate_results(args.output_dir, args.model)


if __name__ == "__main__":
    main()
