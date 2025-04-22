import argparse
import glob
import json
import os
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def check_answer_in_history(history):
    if "\nanswer\n" in history or "</think>" in history:
        return True
    else:
        return False


def process_run(results_filepath: str, details_filepath: str, tokenizer: AutoTokenizer):
    results_json = json.load(open(results_filepath))
    details_df = pd.read_parquet(details_filepath)

    max_new_tokens = results_json["config_general"]["generation_parameters"][
        "max_new_tokens"
    ]
    # Create a new DataFrame with the requested columns
    df = pd.DataFrame(
        columns=[
            "full_prompt",
            "prediction",
            "correct_answer",
            "extracted_answer",
            "correct",
            "truncated",
            "formatted_correctly",
        ]
    )
    # Populate the new DataFrame
    for i, row in details_df.iterrows():
        input_prompt = row["full_prompt"]
        output = row["predictions"][0]
        correct = row["metrics"]["extractive_match"] == 1
        answer = row["gold"][0]
        extracted_answer = row["specifics"]["extracted_predictions"][0]

        full_history = input_prompt + output
        num_tokens = len(tokenizer.tokenize(full_history))

        # Determine if response is truncated
        truncated = num_tokens >= max_new_tokens

        # Determine if answer is formatted correctly
        answer_formatted_correctly = (
            check_answer_in_history(full_history) if not truncated else False
        )

        # Add row to new DataFrame
        df.loc[i] = {
            "full_prompt": input_prompt,
            "prediction": output,
            "correct_answer": answer,
            "extracted_answer": extracted_answer,
            "correct": correct,
            "truncated": truncated,
            "formatted_correctly": answer_formatted_correctly,
        }

    tasks = []
    for task_name in results_json["summary_tasks"].keys():
        tasks.append(task_name)

    assert len(tasks) == 1, "There should only be one task"
    task_name = tasks[0]

    config_task_name = next(iter(results_json["config_tasks"].keys()))

    metric_names = [
        metric["metric_name"]
        for metric in results_json["config_tasks"][config_task_name]["metric"]
    ]

    # Remove metrics that are not in the metric_names list (i.e. _stderr values, which are not needed for aggregation)
    results = results_json["results"][task_name].copy()

    for key in results_json["results"][task_name].keys():
        if key not in metric_names:
            del results[key]

    metadata = {
        "model_name": results_json["config_general"]["model_name"],
        "generation_parameters": results_json["config_general"][
            "generation_parameters"
        ],
        "start_time": results_json["config_general"]["start_time"],
        "end_time": results_json["config_general"]["end_time"],
        "results": results,
        "task_name": config_task_name,
        "task_config": results_json["config_tasks"][config_task_name],
    }
    return metadata, df


def verify_run_metadata(all_results: dict):
    seeds = set()
    seed_to_runs = {}
    # Verify that all runs have the same metadata
    for run_key, run_info in all_results.items():
        # Get the first run key to use as a reference
        first_run_key = next(iter(all_results.keys()))
        # Use deep comparison for task_config dictionaries
        if not json.dumps(
            run_info["metadata"]["task_config"], sort_keys=True
        ) == json.dumps(
            all_results[first_run_key]["metadata"]["task_config"], sort_keys=True
        ):
            raise ValueError(f"Run {run_key} has different task config")

        # Check model name
        if (
            run_info["metadata"]["model_name"]
            != all_results[first_run_key]["metadata"]["model_name"]
        ):
            raise ValueError(f"Run {run_key} has different model name")

        # Check generation parameters
        # Remove seed from generation parameters before comparing
        generation_parameters_1 = run_info["metadata"]["generation_parameters"].copy()
        generation_parameters_2 = all_results[first_run_key]["metadata"][
            "generation_parameters"
        ].copy()

        seed = generation_parameters_1.get("seed")
        seeds.add(seed)

        # Track which runs use the same seed
        if seed not in seed_to_runs:
            seed_to_runs[seed] = []
        seed_to_runs[seed].append(run_key)

        # Remove seed from both dictionaries if present
        if "seed" in generation_parameters_1:
            del generation_parameters_1["seed"]
        if "seed" in generation_parameters_2:
            del generation_parameters_2["seed"]
        if not json.dumps(generation_parameters_1, sort_keys=True) == json.dumps(
            generation_parameters_2, sort_keys=True
        ):
            raise ValueError(f"Run {run_key} has different generation parameters")

    if len(seeds) != len(all_results):
        duplicate_seeds = [seed for seed, runs in seed_to_runs.items() if len(runs) > 1]
        duplicate_runs = {
            seed: runs for seed, runs in seed_to_runs.items() if len(runs) > 1
        }
        raise ValueError(
            f"Duplicate runs found for seeds: {duplicate_seeds}. Duplicate runs: {duplicate_runs}"
        )


def aggregate_results(runs: dict, tokenizer: AutoTokenizer):
    all_results = {}
    for run_key, run_info in runs.items():
        metadata, responses_df = process_run(
            run_info["result_filepath"], run_info["details_filepath"], tokenizer
        )
        all_results[run_key] = {
            "metadata": metadata,
            "responses": responses_df,
        }

    # Aggregate metric values across runs
    metric_values = defaultdict(list)
    for run_key, result in all_results.items():
        metadata = result["metadata"]
        for metric_name, metric_value in metadata["results"].items():
            metric_values[metric_name].append(metric_value)

    # Calculate mean and standard deviation for each metric
    metric_aggregates = {}
    for metric_name, values in metric_values.items():
        mean = np.mean(values)
        std_dev = np.std(values, ddof=0)
        metric_aggregates[metric_name] = {
            "mean": round(mean, 4),
            "std_dev": round(std_dev, 4),
        }

    response_metadata = defaultdict(list)
    for run_key, result in all_results.items():
        responses_df = result["responses"]
        fraction_truncated = responses_df["truncated"].mean()
        fraction_formatted_correctly = responses_df["formatted_correctly"].mean()
        fraction_incorrectly_formatted_non_truncated = (
            1 - fraction_formatted_correctly - fraction_truncated
        )
        response_metadata["truncated_responses"].append(fraction_truncated)
        response_metadata["incorrectly_formatted_non_truncated_responses"].append(
            fraction_incorrectly_formatted_non_truncated
        )
        response_metadata["formatted_correctly_responses"].append(
            fraction_formatted_correctly
        )

    # Calculate mean and standard deviation for each metric
    aggregated_response_metadata = {}
    for metric_name, values in response_metadata.items():
        mean = np.mean(values)
        std_dev = np.std(values, ddof=0)
        aggregated_response_metadata[metric_name] = {
            "mean": round(mean, 4),
            "std_dev": round(std_dev, 4),
        }

    # Extract a reference run for metadata
    reference_run = next(iter(all_results.values()))["metadata"]

    # Collect all seeds
    seeds = {
        result["metadata"]["generation_parameters"]["seed"]
        for result in all_results.values()
    }

    eval_result = {
        "task_name": reference_run["task_name"],
        "num_runs": len(all_results),
        "model_name": reference_run["model_name"],
        "generation_parameters": {
            k: v
            for k, v in reference_run["generation_parameters"].items()
            if k != "seed"
        },
        "seeds": list(seeds),
        "metrics": metric_aggregates,
        "response_metadata": aggregated_response_metadata,
    }

    return eval_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory where lighteval files are stored",
    )
    parser.add_argument(
        "--tokenizer_model",
        default="Qwen/Qwen2.5-32B-Instruct",
        type=str,
        help="Tokenizer to use for aggregation",
    )

    args = parser.parse_args()
    results_dir = args.results_dir
    tokenizer_model = args.tokenizer_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    # Get all the parquet files in the results directory

    # For recursive search, we would use this instead:
    details_files = glob.glob(os.path.join(results_dir, "**/*.parquet"), recursive=True)
    print(f"Found {len(details_files)} parquet files in {results_dir}")

    # Find all JSON files but exclude results_summary.json
    results_files = [
        f
        for f in glob.glob(os.path.join(results_dir, "**/*.json"), recursive=True)
        if not f.endswith("results_summary.json")
    ]
    print(f"Found {len(results_files)} json files in {results_dir}")

    # Group results and details files by key
    runs = {}
    for result_filepath in results_files:
        filename = os.path.basename(result_filepath)
        key = filename.split("results_")[1].split(".")[0]

        # Find the parquet file that contains the key in its filename
        matching_details_filepath = None
        for details_filepath in details_files:
            if key in details_filepath:
                matching_details_filepath = details_filepath
                break

        if matching_details_filepath:
            runs[key] = {
                "result_filepath": result_filepath,
                "details_filepath": matching_details_filepath,
            }
        else:
            print(f"Warning: No matching details file found for {result_filepath}")

    eval_result = aggregate_results(runs, tokenizer)

    # Write the aggregated results to a results_summary.json file
    output_path = os.path.join(results_dir, "results_summary.json")
    with open(output_path, "w") as f:
        json.dump(eval_result, f, indent=4)

    print(f"Results summary written to {output_path}")
    print(f"Results summary:\n{json.dumps(eval_result, indent=4)}")


if __name__ == "__main__":
    main()
