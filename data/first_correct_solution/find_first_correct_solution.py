import asyncio
import os

import pandas as pd
from tqdm import tqdm
from utils import (
    choose_first_correct_solution,
    choose_first_correct_solution_sentence,
    describe_first_correct_solution_location,
    get_context,
    split_into_sentences,
)


async def find_first_correct_solution(question, answer, reasoning):
    location = await describe_first_correct_solution_location(
        question, answer, reasoning
    )
    sentences = split_into_sentences(reasoning)

    excerpts = []
    excerpt_to_sentence_index = {}
    last_added_index = -4  # Initialize to ensure the first found index is always added

    for i, sentence in enumerate(sentences):
        if answer in sentence:
            context = get_context(reasoning, sentence, context_length=500)
            excerpt_to_sentence_index[context] = i
            # Check if this index is far enough from the last added one
            if not excerpts or i - last_added_index >= 4:
                # Add context for this new, sufficiently spaced index
                excerpts.append(context)
                last_added_index = i
            else:
                # This index is too close to the last one, replace the last context
                excerpts[-1] = context  # Replace the last element
                last_added_index = (
                    i  # Update the index of the last added/updated excerpt
                )
    # print(
    #     f"Found {len(excerpts)} potential first correct solution excerpts after merging"
    # )
    if len(excerpts) == 1:
        index = 0
    else:
        index = await choose_first_correct_solution(
            question, answer, reasoning, location, excerpts
        )
    excerpt = excerpts[index]
    sentence_index = excerpt_to_sentence_index[excerpt]

    last_added_index = sentence_index
    candidate_sentences = []
    for i in range(sentence_index, -1, -1):
        sentence = sentences[i]
        if last_added_index - i > 4:
            break
        if answer in sentence:
            candidate_sentences.append(sentence)
            last_added_index = i

    candidate_sentences = candidate_sentences[::-1]

    if len(candidate_sentences) > 1:
        # print(f"Choosing from {len(candidate_sentences)} sentences")
        index = await choose_first_correct_solution_sentence(
            question, answer, reasoning, excerpt, candidate_sentences
        )
        sentence_index = sentences.index(candidate_sentences[index])
    else:
        sentence_index = sentences.index(candidate_sentences[0])

    sentence = sentences[sentence_index]
    # print(f"Answer: {answer}")
    # print(f"First correct solution sentence: {sentence}")

    return reasoning[: reasoning.find(sentence) + len(sentence)]


async def main(csv_path, save_path, max_concurrent):
    # Initialize or load existing progress
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.read_csv(csv_path)

        # Add a 'First Correct Solution' column if it doesn't exist
        if "First Correct Solution" not in df.columns:
            df["First Correct Solution"] = None

            print("Added 'First Correct Solution' column to the dataframe")

    # Create a lock for thread-safe dataframe updates
    df_lock = asyncio.Lock()
    # Create a semaphore to rate limit concurrent API calls
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_row(i, row, progress_bar):
        # Use the semaphore to limit concurrent API calls
        async with semaphore:
            fcs = await find_first_correct_solution(
                question=row["Question"],
                answer=str(int(row["Answer Attempt"])),
                reasoning=row["Reasoning"],
            )

        # Use the lock when updating the dataframe
        async with df_lock:
            df.loc[i, "First Correct Solution"] = fcs
            df.to_csv(save_path, index=False)

        # Update progress bar
        progress_bar.update(1)

        return i

    # Create tasks for all rows we want to process
    rows_to_process = []
    for i, row in df.iterrows():
        # Convert 'Correct' to boolean if it's not already, and skip if not correct or already processed
        # correct = row["Correct"]
        # if isinstance(correct, str):
        #     correct = correct.lower() == "true"
        if not pd.isna(row["First Correct Solution"]):
            continue

        rows_to_process.append((i, row))

    # Process all tasks concurrently
    if rows_to_process:
        print(
            f"Processing {len(rows_to_process)} rows concurrently (max 20 at a time)..."
        )
        with tqdm(total=len(rows_to_process), desc="Processing rows") as progress_bar:
            tasks = [process_row(i, row, progress_bar) for i, row in rows_to_process]
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect reasoning traces")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/results/aime_1983_2023_qwq-32b_traces.csv",
        help="Path to the CSV file containing the traces",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="data/results/aime_1983_2023_qwq-32b_fcs_traces.csv",
        help="Path to save the CSV file containing the first correct solution traces",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Maximum number of concurrent API calls",
    )
    args = parser.parse_args()
    asyncio.run(main(args.csv_path, args.save_path, args.max_concurrent))
