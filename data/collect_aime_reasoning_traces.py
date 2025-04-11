import asyncio
import csv
import json
import os
from collections import defaultdict
from io import StringIO
from random import random
from typing import Tuple

import aiofiles
import pandas as pd
from tqdm import tqdm
from utils.constants import (
    ANSWER_EXTRACTION_PROMPT,
    CLIENTS,
    GRADING_PROMPT,
    OPENAI_CLIENT,
)
from utils.response_processing import (
    process_claude_response,
    process_think_tagged_output,
)

# Change working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


async def verify_answer_correctness(
    question: str,
    answer: str,
    answer_attempt: str,
) -> bool:
    """
    Verify the correctness of the answer.

    Args:
        question: The question to verify the answer for.
        attempt: The attempt to verify the answer for.
        solution: The correct answer to the question.

    Returns:
        A tuple containing the attempt answer, correctness of the answer, and the explanation.
    """
    response = await OPENAI_CLIENT.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": GRADING_PROMPT,
            },
            {
                "role": "user",
                "content": f"# Question\n{question}\n\n# Correct answer\n{answer}\n\n# Student's answer\n{answer_attempt}",
            },
        ],
        response_format={"type": "json_object"},
    )
    json_response = json.loads(response.choices[0].message.content)
    correct = json_response["correct"]
    if isinstance(correct, bool):
        return correct
    else:
        raise ValueError(f"Unexpected response format: {type(correct)}")


async def extract_answer(attempt: str) -> str:
    """
    Extract the answer from a student's attempt at a problem.
    """
    response = await OPENAI_CLIENT.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": ANSWER_EXTRACTION_PROMPT,
            },
            {
                "role": "user",
                "content": f"# Attempt\n{attempt}",
            },
        ],
        response_format={"type": "json_object"},
    )
    json_response = json.loads(response.choices[0].message.content)
    extracted_answer = json_response["extracted_answer"]

    extracted_answer = str(extracted_answer)

    # Remove commas, spaces, and other formatting characters
    extracted_answer = extracted_answer.replace(",", "").strip()

    # Remove any surrounding characters like brackets, quotes, etc.
    extracted_answer = extracted_answer.strip("[](){}<>\"'")

    return extracted_answer


async def generate_response(question: str, model: str) -> Tuple[str, str]:
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
            + "\n\nPlease reason step by step, and put your final answer within \boxed{}."
        )

    match model:
        case "claude-3-7":
            message = await client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=128000,
                thinking={"type": "enabled", "budget_tokens": 64000},
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
                max_tokens=128000,
                model="accounts/fireworks/models/deepseek-r1",
            )
            return process_think_tagged_output(response.choices[0].message.content)

        case "qwq-32b":
            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": question}],
                temperature=0.6,
                top_p=0.95,
                max_tokens=128000,
                model="accounts/fireworks/models/qwq-32b",
            )
            return process_think_tagged_output(response.choices[0].message.content)

        case _:
            raise ValueError(f"Unsupported model type: {model}")


async def generate_correct_response(
    question: str, answer: str, model: str, max_attempts: int = 16
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
            reasoning, attempt = await generate_response(question, model)
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
        except Exception as e:
            print(f"Attempt {attempt_num + 1} failed with error: {str(e)}")
            continue

    # If we've exhausted all attempts, return the best response (first one as fallback)
    if not responses:
        raise ValueError(
            f"Failed to generate any valid responses after {max_attempts} attempts"
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
) -> bool:
    """Process a single question with semaphore control."""
    async with api_semaphore:
        try:
            tqdm.write(f"Processing question {row['ID']}...")
            # Sleep for a random time between 1 and 10 seconds to avoid rate limiting on request rate
            await asyncio.sleep(random() * 10)
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


async def main(model_name: str, max_concurrent: int, max_attempts: int):
    """Main function to process all questions."""
    os.makedirs("results", exist_ok=True)
    output_file = f"results/aime_1983_2023_{model_name}_traces.csv"

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

    df = pd.read_csv(
        "hf://datasets/di-zhang-fdu/AIME_1983_2024/AIME_Dataset_1983_2024.csv"
    )
    df = df[df["Year"] < 2024]
    df.to_csv(
        "aime_1983_2023.csv",
        index=False,
    )

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
        default="qwq-32b",
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
        "--max-attempts",
        type=int,
        default=64,
        help="Maximum number of attempts to generate a correct response",
    )
    args = parser.parse_args()

    asyncio.run(main(args.model, args.max_concurrent, args.max_attempts))
