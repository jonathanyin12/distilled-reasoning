import asyncio
import csv
from io import StringIO

import aiofiles


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
    """
    Save a result to the output CSV file with proper locking.

    Args:
        output_file: Path to the output CSV file
        file_lock: Lock for synchronizing file access
        index: Question index
        question: The question text
        reasoning: Model's reasoning
        attempt: Model's attempt
        attempt_answer: Model's answer
        explanation: Grading explanation
        is_correct: Whether the answer is correct
        solution: Correct solution
        cot_type: Type of chain-of-thought
        source_type: Source of the question
        metadata: Additional metadata
    """
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
