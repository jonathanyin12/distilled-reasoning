import os

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# Prompts
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
