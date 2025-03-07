import asyncio
import os
from typing import Tuple

import aiofiles
import pandas as pd
from tqdm import tqdm
from utils.constants import CLIENTS
from utils.core import generate_verified_response
from utils.file_ops import save_result

# Change working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


async def process_question(
    index: int,
    row: pd.Series,
    model_name: str,
    output_file: str,
    file_lock: asyncio.Lock,
    api_semaphore: asyncio.Semaphore,
) -> Tuple[int, bool]:
    """Process a single question with semaphore control."""
    async with api_semaphore:
        try:
            tqdm.write(f"Processing question {index}...")
            (
                reasoning,
                attempt,
                attempt_answer,
                is_correct,
                explanation,
            ) = await generate_verified_response(
                row["question"],
                row["cot_type"],
                row["solution"],
                model_name,
            )

            await save_result(
                output_file=output_file,
                file_lock=file_lock,
                index=index,
                **row.to_dict(),  # Pass all row data
                reasoning=reasoning,
                attempt=attempt,
                attempt_answer=attempt_answer,
                explanation=explanation,
                is_correct=is_correct,
            )
            return index, True
        except Exception as e:
            print(f"Error processing question {index}: {e}")
            return index, False


async def main(model_name: str, max_concurrent: int):
    """Main function to process all questions."""
    os.makedirs("results", exist_ok=True)
    output_file = f"results/s1k_{model_name}.csv"

    # Initialize or load existing progress
    processed_indices = set()
    if os.path.exists(output_file):
        try:
            processed_indices = set(pd.read_csv(output_file)["index"].astype(int))
            print(f"Found {len(processed_indices)} already processed questions.")
        except Exception as e:
            print(f"Error reading existing file: {e}")
    else:
        headers = "index,solution,question,cot_type,source_type,metadata,reasoning,attempt,attempt_answer,matches_solution,grading_explanation\n"
        async with aiofiles.open(output_file, "w") as f:
            await f.write(headers)

    df = pd.read_csv("s1k_questions.csv")

    semaphore = asyncio.Semaphore(max_concurrent)
    lock = asyncio.Lock()
    tasks = [
        asyncio.create_task(
            process_question(
                i,
                row,
                model_name,
                output_file,
                lock,
                semaphore,
            )
        )
        for i, (_, row) in enumerate(df.iterrows())
        if i not in processed_indices
    ]

    if tasks:
        print(f"{len(tasks)} tasks to process...")
        with tqdm(total=len(tasks), desc="Processing questions") as progress_bar:
            results = await asyncio.gather(
                *(update_progress(task, progress_bar) for task in tasks)
            )

        successes = sum(1 for _, success in results if success)
        print(f"Processed {len(results)} questions with {successes} successes.")
    else:
        print("No new questions to process.")


async def update_progress(task, progress_bar):
    """Helper function to update progress bar."""
    result = await task
    progress_bar.update(1)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate correct reasoning traces")
    parser.add_argument(
        "--model",
        type=str,
        default="qwq-32b",
        choices=list(CLIENTS.keys()),
        help="Model to use for generation",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum number of concurrent API calls",
    )
    args = parser.parse_args()

    asyncio.run(main(args.model, args.max_concurrent))
