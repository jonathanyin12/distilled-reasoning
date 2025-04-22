import json
from typing import Tuple

from utils.clients import OPENAI_CLIENT


def process_think_tagged_output(output: str) -> Tuple[str, str]:
    """
    Process output with <think> tags. Used for R1 and QwQ.

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
        raise ValueError("Unexpected output format. Closing think tag not found.")
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


def process_grok_response(response) -> Tuple[str, str]:
    """
    Process Grok 3 message response.

    Args:
        message: The message response from Grok.

    Returns:
        A tuple containing the reasoning and answer.
    """
    reasoning = response.choices[0].message.reasoning_content
    answer = response.choices[0].message.content

    if not answer:
        raise ValueError("No answer text found in Grok response")
    if not reasoning:
        raise ValueError("No reasoning text found in Grok response")

    return reasoning, answer


GRADING_PROMPT = """You are an AI assistant tasked with grading a student's answer to a question. Your job is to judge whether the attempt is correct by comparing it with the correct answer.

The user will provide the question, the student's answer and the correct answer in the following format:

# Question
{question}

# Correct answer
{answer}

# Student's answer
{attempt}

Output your response in JSON format:
{
    "correct": <boolean indicating if the student's answer is correct>
}"""


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


ANSWER_EXTRACTION_PROMPT = """You are an AI assistant tasked with extracting the answer from a student's attempt at a problem. 

The user will provide you with their attempt at a problem in the following format:

# Attempt
{attempt}

Your job is to extract the answer from the attempt. Typically the answer will be boxed. In such cases, extract the answer from the boxed text.


Output the following in JSON format:
{
    "extracted_answer": <the extracted answer>
}"""


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

    extracted_answer = str(extracted_answer).strip()

    return extracted_answer
