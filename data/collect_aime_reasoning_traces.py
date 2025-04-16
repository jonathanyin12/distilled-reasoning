import asyncio
import csv
import os
from collections import defaultdict
from io import StringIO
from random import random
from typing import Tuple

import aiofiles
import pandas as pd
from tqdm import tqdm
from utils.constants import (
    CLIENTS,
)
from utils.response_processing import (
    extract_answer,
    process_claude_response,
    process_grok_response,
    process_think_tagged_output,
    verify_answer_correctness,
)

# Change working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


async def generate_response(
    question: str, model: str, max_tokens: int = 32768
) -> Tuple[str, str]:
    """
    Generate a response from the model.

    Args:
        question: The question to generate a response for.
        question_type: The type of question to generate a response for.
        model: The model to generate a response for.

    Returns:
        A tuple containing the reasoning and answer.
    """
    if model not in CLIENTS:
        raise ValueError(f"Model {model} not supported")

    client = CLIENTS[model]

    # Format math questions for applicable models
    if model in ["deepseek-r1", "qwq-32b"]:
        question = (
            question
            + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
    elif "grok-3-mini" in model:
        question = (
            question
            + "\n\nPlease reason step by step, and put your final answer within \\boxed{}. Your final answer should not contain leading zeros."
        )

    match model:
        case "claude-3-7":
            message = await client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_completion_tokens=max_tokens,
                thinking={"type": "enabled", "budget_tokens": max_tokens},
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": question}],
                    }
                ],
                betas=["output-128k-2025-02-19"],
            )
            return process_claude_response(message)

        case "deepseek-r1":
            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": question}],
                temperature=0.6,
                max_completion_tokens=max_tokens,
                model="accounts/fireworks/models/deepseek-r1",
            )
            return process_think_tagged_output(response.choices[0].message.content)

        case "qwq-32b":
            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": question}],
                temperature=0.6,
                top_p=0.95,
                max_completion_tokens=max_tokens,
                model="accounts/fireworks/models/qwq-32b",
            )
            return process_think_tagged_output(response.choices[0].message.content)
        # case "grok-3-mini-low": # ISSUE: has leading zeros in the answer
        #     response = await client.chat.completions.create(
        #         messages=[{"role": "user", "content": question}],
        #         reasoning_effort="low",
        #         model="grok-3-mini",
        #         max_completion_tokens=32768,
        #     )
        #     if response.choices[0].finish_reason == "length":
        #         raise ValueError("Terminated due to length limit")
        #     return process_grok_response(response)
        case "grok-3-mini-high":
            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": question}],
                reasoning_effort="high",
                model="grok-3-mini",
                max_completion_tokens=max_tokens,
            )
            if response.choices[0].finish_reason == "length":
                raise ValueError("Terminated due to length limit")
            return process_grok_response(response)
        case _:
            raise ValueError(f"Unsupported model type: {model}")


async def generate_correct_response(
    question: str,
    answer: str,
    model: str,
    max_attempts: int = 16,
    majority_voting: bool = False,
    max_tokens: int = 32768,
) -> Tuple[str, str, bool, str]:
    """
    Generate a correct response from the model.

    Try again up to max_attempts times to generate a response that is verified to be correct.
    If none of the attempts are verified to be correct, return the majority vote answer.

    Args:
        question: The question to generate a response for.
        answer: The correct answer to the question.
        model: The model to generate a response for.
        max_attempts: The maximum number of attempts to generate a correct response.

    Returns:
        A tuple containing the reasoning, solution attempt, answer attempt, and correctness.
    """
    responses = []
    for attempt_num in range(max_attempts):
        try:
            reasoning, attempt = await generate_response(question, model, max_tokens)
            answer_attempt = await extract_answer(attempt)
            correct = await verify_answer_correctness(
                question=question, answer=answer, answer_attempt=answer_attempt
            )
            responses.append(
                {
                    "reasoning": reasoning,
                    "solution_attempt": attempt,
                    "answer_attempt": answer_attempt,
                    "correct": correct,
                }
            )

            # If the answer is correct, return it immediately
            if correct:
                return reasoning, attempt, answer_attempt, correct
            else:
                print(
                    f"Attempt {attempt_num + 1} failed. Correct answer: {answer}. Attempted answer: {answer_attempt}"
                )
        except Exception as e:
            print(f"Attempt {attempt_num + 1} failed with error: {str(e)}")
            continue

    # If we've exhausted all attempts, return the best response (first one as fallback)
    if not responses:
        raise ValueError(
            f"Failed to generate any valid responses after {max_attempts} attempts"
        )
    elif not majority_voting:
        raise ValueError(
            f"Failed to generate any correct responses after {max_attempts} attempts"
        )
    else:
        # Implement majority voting mechanism based on attempt_answer
        answer_counts = defaultdict(int)
        for response in responses:
            answer_counts[response["answer_attempt"]] += 1

        # Find the most common answer
        majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
        print(
            f"Used majority voting to get answer. Answers: {answer_counts}. Majority answer: {majority_answer}"
        )
        # Find the first response with the majority answer
        for response in responses:
            if response["answer_attempt"] == majority_answer:
                return (
                    response["reasoning"],
                    response["solution_attempt"],
                    response["answer_attempt"],
                    response["correct"],
                )


async def process_question(
    row: pd.Series,
    model_name: str,
    output_file: str,
    file_lock: asyncio.Lock,
    api_semaphore: asyncio.Semaphore,
    max_attempts: int,
    majority_voting: bool,
    max_tokens: int,
) -> bool:
    """Process a single question with semaphore control."""
    async with api_semaphore:
        try:
            tqdm.write(f"Processing question {row['ID']}...")
            # Sleep for a random time between 1 and 10 seconds to avoid rate limiting on request rate
            await asyncio.sleep(random() * 20)
            (
                reasoning,
                solution_attempt,
                answer_attempt,
                correct,
            ) = await generate_correct_response(
                row["Question"],
                row["Answer"],
                model_name,
                max_attempts,
                majority_voting,
                max_tokens,
            )

            async with file_lock:  # Only one task can write at a time
                async with aiofiles.open(output_file, "a") as f:
                    output = StringIO()
                    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

                    writer.writerow(
                        [
                            row["ID"],
                            row["Year"],
                            row["Problem Number"],
                            row["Question"],
                            row["Answer"],
                            row["Part"],
                            reasoning,
                            solution_attempt,
                            answer_attempt,
                            correct,
                        ]
                    )

                    await f.write(output.getvalue())

            return True
        except Exception as e:
            print(f"Error processing question {row['ID']}: {e}")
            return False


async def main(
    model_name: str,
    max_concurrent: int,
    max_attempts: int,
    majority_voting: bool,
    max_tokens: int,
):
    """Main function to process all questions."""
    os.makedirs("results", exist_ok=True)
    output_file = f"aime_1983_2023_{model_name}_traces_{max_tokens}.csv"

    # Initialize or load existing progress
    processed_indices = set()
    if os.path.exists(output_file):
        try:
            processed_indices = set(pd.read_csv(output_file)["ID"])
            print(f"Found {len(processed_indices)} already processed questions.")
        except Exception as e:
            print(f"Error reading existing file: {e}")
    else:
        headers = "ID,Year,Problem Number,Question,Answer,Part,Reasoning,Solution Attempt,Answer Attempt,Correct\n"
        async with aiofiles.open(output_file, "w") as f:
            await f.write(headers)

    # df = pd.read_csv(
    #     "hf://datasets/di-zhang-fdu/AIME_1983_2024/AIME_Dataset_1983_2024.csv"
    # )
    # df = df[df["Year"] < 2024]
    # df.to_csv(
    #     "aime_1983_2023.csv",
    #     index=False,
    # )

    df = pd.read_csv("aime_1983_2023.csv")

    semaphore = asyncio.Semaphore(max_concurrent)
    lock = asyncio.Lock()
    tasks = [
        asyncio.create_task(
            process_question(
                row,
                model_name,
                output_file,
                lock,
                semaphore,
                max_attempts,
                majority_voting,
                max_tokens,
            )
        )
        for _, row in df.iterrows()
        if row["ID"] not in processed_indices
    ]

    if tasks:
        print(f"{len(tasks)} tasks to process...")
        with tqdm(total=len(tasks), desc="Processing questions") as progress_bar:
            results = await asyncio.gather(
                *(update_progress(task, progress_bar) for task in tasks)
            )

        successes = sum(1 for result in results if result)
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

    parser = argparse.ArgumentParser(description="Collect reasoning traces")
    parser.add_argument(
        "--model",
        type=str,
        default="grok-3-mini-high",
        choices=list(CLIENTS.keys()),
        help="Model to use for generation",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Maximum number of concurrent API calls",
    )
    parser.add_argument(
        "--majority-voting",
        type=bool,
        default=True,
        help="Use majority voting to get the answer",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Maximum number of attempts to generate a correct response",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            args.model,
            args.max_concurrent,
            args.max_attempts,
            args.majority_voting,
            args.max_tokens,
        )
    )
