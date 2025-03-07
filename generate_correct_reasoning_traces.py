import json
import os
import aiofiles
import asyncio
import pandas as pd
from typing import Tuple
from dotenv import load_dotenv
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from tqdm import tqdm
import csv
from io import StringIO

load_dotenv()

# Constants
GRADING_PROMPT = """
You are an AI assistant tasked with grading a student's attempt at a problem. The user will provide you with the question itself, an attempt made by a student and the correct answer to the problem. Your job is to judge whether the attempt is correct by comparing it with the correct answer. If the expected solution concludes with a number or choice, there should be no ambiguity. If the expected solution involves going through the entire reasoning process, you should judge the attempt based on whether the reasoning process is correct with correct answer if helpful.

The user will provide the attempt and the correct answer in the following format:

# Problem
{problem}

## Attempt
{attempt}

## Correct answer
{solution}

Explain your reasoning and output True if the attempt is correct, False otherwise. Output in JSON format:
{
    "attempt_answer": "The answer to the question from the attempt. This should be a number or choice if applicable. If there are multiple parts to the question, attempt_answer should be a list of answers.",
    "explanation": "Your explanation here",
    "correct": True | False,
}"""

MATCHING_PROMPT = """You are an AI assistant tasked with determining if an answer is equivalent to any of the answers in a list of answers. If the answer is equivalent to any of the answers in the list, return the index of the matching answer. If it is not, return -1.


The user will provide you with the list of answers and the answer you are trying to match in the following format:

# List of answers
{answers}

# Answer to match
{answer}

Output in JSON format:
{
    "answer_matched": True | False,
    "matching_answer_index": "The index (1-indexed) of the matching answer from the list of answers. If there is no matching answer, return -1.",
}"""


# Client configuration
CLIENTS = {
    "claude-3-7": AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
    "deepseek-r1": AsyncOpenAI(
        api_key=os.getenv("FIREWORKS_API_KEY"),
        base_url="https://api.fireworks.ai/inference/v1",
    ),
    "qwq-32b": AsyncOpenAI(
        api_key=os.getenv("FIREWORKS_API_KEY"),
        base_url="https://api.fireworks.ai/inference/v1",
    ),
}
OPENAI_CLIENT = AsyncOpenAI()


# Response processing utilities
def process_think_tagged_output(output: str) -> Tuple[str, str]:
    """
    Process output with <think> tags.

    Args:
        output: The raw output from the model.

    Returns:
        A tuple containing the reasoning and answer.
    """
    output = output.replace("<think>", "")  # Remove opening think tag
    if len(output.split("</think>")) == 2:
        reasoning = output.split("</think>")[0]
        answer = output.split("</think>")[1]
    else:
        raise ValueError(f"Unexpected output format. Output: {output}")
    return reasoning.strip(), answer.strip()


def process_claude_response(message) -> Tuple[str, str]:
    """
    Process Claude 3.7 message response.

    Args:
        message: The message response from Claude.

    Returns:
        A tuple containing the reasoning and answer.
    """
    thinking_block = next((b for b in message.content if b.type == "thinking"), None)
    text_block = next((b for b in message.content if b.type == "text"), None)

    if not text_block:
        raise ValueError("No answer text found in Claude response")

    if not thinking_block:
        raise ValueError("No thinking block found in Claude response")

    if any(b.type == "redacted_thinking" for b in message.content):
        raise ValueError("Redacted thinking detected.")

    return thinking_block.thinking, text_block.text


# Core functionality
async def find_matching_answer(answers: list[str], attempt_answer: str):
    """
    Find the matching answer from a list of answers. If none of the answers match, return the attempt answer.
    """
    if len(answers) == 0:
        return attempt_answer

    response = await OPENAI_CLIENT.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": MATCHING_PROMPT,
            },
            {
                "role": "user",
                "content": "# List of answers\n"
                + "\n".join([f"{i + 1}. {answer}" for i, answer in enumerate(answers)])
                + f"\n\n# Answer to match\n{attempt_answer}",
            },
        ],
        response_format={"type": "json_object"},
    )
    json_response = json.loads(response.choices[0].message.content)

    if json_response["answer_matched"]:
        return answers[json_response["matching_answer_index"] - 1]
    else:
        return attempt_answer


async def verify_answer_correctness(
    question: str, attempt: str, solution: str
) -> Tuple[bool, str, str]:
    """
    Verify the correctness of the answer.

    Args:
        question: The question to verify the answer for.
        attempt: The attempt to verify the answer for.
        solution: The correct answer to the question.

    Returns:
        A tuple containing the correctness of the answer and the explanation.
    """
    response = await OPENAI_CLIENT.chat.completions.create(
        model="o3-mini",
        messages=[
            {
                "role": "system",
                "content": GRADING_PROMPT,
            },
            {
                "role": "user",
                "content": f"# Problem\n{question}\n\n## Attempt\n{attempt}\n\n## Correct answer\n{solution}",
            },
        ],
        response_format={"type": "json_object"},
    )
    json_response = json.loads(response.choices[0].message.content)

    # Ensure attempt_answer is a string
    if isinstance(json_response["attempt_answer"], (int, float, bool)):
        attempt_answer = str(json_response["attempt_answer"])
    else:
        attempt_answer = json.dumps(json_response["attempt_answer"])

    return (
        json_response["correct"],
        json_response["explanation"],
        attempt_answer,
    )


async def generate_response(
    question: str, question_type: str, model: str
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
    # Format question if it's a math question for applicable models
    if question_type == "math" and model in ["deepseek-r1", "qwq-32b"]:
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


async def generate_verified_response(
    question: str, question_type: str, solution: str, model: str, max_attempts: int = 64
) -> Tuple[str, str, bool, str]:
    """
    Generate a correct response from the model.

    Try again up to max_attempts times to generate a response that is verified to be correct.
    If none of the attempts are verified to be correct, return the majority vote answer.

    Args:
        question: The question to generate a response for.
        question_type: The type of question to generate a response for.
        solution: The correct answer to the question.
        model: The model to generate a response for.
        max_attempts: The maximum number of attempts to generate a correct response.

    Returns:
        A tuple containing the reasoning, answer, correctness, and grading explanation.
    """
    responses = []
    for attempt_num in range(max_attempts):
        try:
            reasoning, attempt = await generate_response(question, question_type, model)
            is_correct, explanation, attempt_answer = await verify_answer_correctness(
                question, attempt, solution
            )
            responses.append(
                {
                    "reasoning": reasoning,
                    "attempt": attempt,
                    "attempt_answer": attempt_answer,
                    "is_correct": is_correct,
                    "explanation": explanation,
                }
            )

            # If the answer is correct, return it immediately
            if is_correct:
                return reasoning, attempt, attempt_answer, is_correct, explanation
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
        answer_counts = {}
        for response in responses:
            answer = await find_matching_answer(
                list(answer_counts.keys()), response["attempt_answer"]
            )
            if answer in answer_counts:
                answer_counts[answer] += 1
            else:
                answer_counts[answer] = 1

        # Find the most common answer
        majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
        print(
            f"Used majority voting to get answer. Answers: {answer_counts}. Majority answer: {majority_answer}"
        )
        # Find the first response with the majority answer
        for response in responses:
            if response["attempt_answer"] == majority_answer:
                return (
                    response["reasoning"],
                    response["attempt"],
                    response["attempt_answer"],
                    response["is_correct"],
                    response["explanation"],
                )
        raise ValueError(
            f"Failed to generate any valid responses after {max_attempts} attempts"
        )


# File operations
async def save_result(
    output_file: str,
    file_lock: asyncio.Lock,
    index: int,
    question: str,
    reasoning: str,
    attempt: str,
    attempt_answer: str,
    explanation: str,
    is_correct: bool,
    solution: str = "",
    cot_type: str = "",
    source_type: str = "",
    metadata: str = "",
):
    async with file_lock:  # Only one task can write at a time
        async with aiofiles.open(output_file, "a") as f:
            output = StringIO()
            writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

            writer.writerow(
                [
                    index,
                    solution,
                    question,
                    cot_type,
                    source_type,
                    metadata,
                    reasoning,
                    attempt,
                    attempt_answer,
                    is_correct,
                    explanation,
                ]
            )

            await f.write(output.getvalue())


# Question processing
async def process_question(
    index: int,
    question: str,
    solution: str,
    question_type: str,
    model_name: str,
    output_file: str,
    file_lock: asyncio.Lock,
    api_semaphore: asyncio.Semaphore,
    source_type: str = "",
    metadata: str = "",
) -> Tuple[int, bool]:
    """
    Process a single question with semaphore control.

    Returns:
        A tuple containing the index and success status.
    """
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
                question, question_type, solution, model_name, max_attempts=2
            )

            # Save the result to the CSV file with file lock
            await save_result(
                output_file=output_file,
                file_lock=file_lock,
                index=index,
                question=question,
                reasoning=reasoning,
                attempt=attempt,
                attempt_answer=attempt_answer,
                explanation=explanation,
                is_correct=is_correct,
                solution=solution,
                cot_type=question_type,
                source_type=source_type,
                metadata=metadata,
            )
            return index, True
        except Exception as e:
            print(f"Error processing question {index}: {e}")
            return index, False


async def main(model_name: str, max_concurrent: int):
    """
    Main function to process all questions.

    Args:
        model_name: The model to use.
        max_concurrent: The maximum number of concurrent API calls.
    """
    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Use consistent filename without timestamp
    output_file = f"results/s1k_{model_name}.csv"

    # Check if file exists and load processed indices
    processed_indices = set()
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            processed_indices = set(existing_df["index"].astype(int).tolist())
            print(f"Found {len(processed_indices)} already processed questions.")
        except Exception as e:
            print(f"Error reading existing file: {e}")
    else:
        # Create the CSV file with headers if it doesn't exist
        async with aiofiles.open(output_file, "w") as f:
            # Include original columns plus new columns
            await f.write(
                "index,solution,question,cot_type,source_type,metadata,reasoning,attempt,attempt_answer,matches_solution,grading_explanation\n"
            )

    df = pd.read_csv("s1k_questions.csv")

    api_semaphore = asyncio.Semaphore(max_concurrent)
    file_lock = asyncio.Lock()

    # Create tasks for all questions that haven't been processed yet
    tasks = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i not in processed_indices:
            question = row["question"]
            solution = row["solution"]
            question_type = row["cot_type"]
            source_type = row.get("source_type", "")
            metadata = row.get("metadata", "")

            task = asyncio.create_task(
                process_question(
                    i,
                    question,
                    solution,
                    question_type,
                    model_name,
                    output_file,
                    file_lock,
                    api_semaphore,
                    source_type,
                    metadata,
                )
            )
            tasks.append(task)
        else:
            print(f"Skipping already processed question index {i}")

    # Process questions concurrently
    if tasks:
        print(f"{len(tasks)} tasks to process...")
        progress_bar = tqdm(total=len(tasks), desc="Processing questions")

        # Define a callback to update the progress bar
        async def update_progress(task):
            result = await task
            progress_bar.update(1)
            return result

        # Wrap each task with the progress callback
        wrapped_tasks = [update_progress(task) for task in tasks]
        results = await asyncio.gather(*wrapped_tasks)
        progress_bar.close()

        successes = sum(1 for _, success in results if success)
        print(f"Processed {len(results)} questions with {successes} successes.")
    else:
        print("No new questions to process.")


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
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
    model_name = args.model
    max_concurrent = args.max_concurrent

    asyncio.run(main(model_name, max_concurrent))
