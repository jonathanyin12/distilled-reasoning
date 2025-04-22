import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def check_answer_in_history(history):
    if "\nanswer\n" in history or "</think>" in history:
        return True
    else:
        return False


def process_run_results_json(path: str):
    results_json = json.load(open(path))
    task_names = []
    results = {}
    for task_name in results_json["summary_tasks"].keys():
        task_names.append(task_name)
        task_result_dict = results_json["results"][task_name].copy()

        # Remove metrics that are not in the metric_names list (i.e. _stderr values, which are not needed for aggregation)
        config_task_name = "|".join(task_name.split("|")[:-1])
        task_metric_names = [
            metric["metric_name"]
            for metric in results_json["config_tasks"][config_task_name]["metric"]
        ]
        for key in results_json["results"][task_name].keys():
            if key not in task_metric_names:
                del task_result_dict[key]
        results[task_name] = task_result_dict

    seed = results_json["config_general"]["generation_parameters"]["seed"]
    del results_json["config_general"]["generation_parameters"]["seed"]
    metadata = {
        "model_name": results_json["config_general"]["model_name"],
        "seed": seed,
        "generation_parameters": results_json["config_general"][
            "generation_parameters"
        ],
        "start_time": results_json["config_general"]["start_time"],
        "end_time": results_json["config_general"]["end_time"],
        "results": results,
        "task_names": task_names,
        "task_config": results_json["config_tasks"],
    }
    return metadata


def process_run(run_info: dict, tokenizer: AutoTokenizer):
    metadata = process_run_results_json(run_info["result_filepath"])

    task_names = metadata["task_names"]

    response_dfs = {}
    for details_filepath in run_info["details_filepaths"]:
        for task_name in task_names:
            if task_name in details_filepath:
                details_df = pd.read_parquet(details_filepath)

                max_new_tokens = metadata["generation_parameters"]["max_new_tokens"]
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
                    answer = row["gold"][0]

                    # This assumes that the metric is the extractive match metric
                    correct = row["metrics"]["extractive_match"] == 1
                    extracted_answer = row["specifics"]["extracted_predictions"][0]

                    full_history = input_prompt + output
                    num_tokens = len(tokenizer.tokenize(full_history))

                    # Determine if response is truncated
                    truncated = num_tokens >= max_new_tokens

                    # Determine if answer is formatted correctly
                    answer_formatted_correctly = (
                        check_answer_in_history(full_history)
                        if not truncated
                        else False
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
                response_dfs[task_name] = df
    return metadata, response_dfs


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
        if not json.dumps(
            run_info["metadata"]["generation_parameters"], sort_keys=True
        ) == json.dumps(
            all_results[first_run_key]["metadata"]["generation_parameters"],
            sort_keys=True,
        ):
            raise ValueError(f"Run {run_key} has different generation parameters")

        seed = run_info["metadata"]["seed"]
        seeds.add(seed)

        # Track which runs use the same seed
        if seed not in seed_to_runs:
            seed_to_runs[seed] = []
        seed_to_runs[seed].append(run_key)

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
        metadata, responses_dfs = process_run(run_info, tokenizer)
        all_results[run_key] = {
            "metadata": metadata,
            "responses": responses_dfs,
        }

    verify_run_metadata(all_results)

    reference_run = next(iter(all_results.values()))["metadata"]
    seeds = {result["metadata"]["seed"] for result in all_results.values()}
    tasks = reference_run["task_names"]
    eval_time = np.mean(
        [
            result["metadata"]["end_time"] - result["metadata"]["start_time"]
            for result in all_results.values()
        ]
    )
    eval_result = {
        "tasks": tasks,
        "num_runs": len(all_results),
        "model_name": reference_run["model_name"],
        "generation_parameters": reference_run["generation_parameters"],
        "seeds": list(seeds),
        "average_evaluation_seconds": eval_time,
    }

    for task in tasks:
        # Aggregate metric values across runs
        metric_values = defaultdict(list)
        for run_key, result in all_results.items():
            metadata = result["metadata"]
            for metric_name, metric_value in metadata["results"][task].items():
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
            responses_df = result["responses"][task]
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

        eval_result[task] = {
            "metrics": metric_aggregates,
            "response_metadata": aggregated_response_metadata,
        }

    return eval_result


def main(results_dir: str, tokenizer_model: str):
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
        matching_details_filepaths = []
        for details_filepath in details_files:
            if key in details_filepath:
                matching_details_filepaths.append(details_filepath)

        if matching_details_filepaths:
            runs[key] = {
                "result_filepath": result_filepath,
                "details_filepaths": matching_details_filepaths,
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
    main(results_dir, tokenizer_model)
