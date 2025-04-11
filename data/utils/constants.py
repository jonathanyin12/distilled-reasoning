import os

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI

load_dotenv()

# Prompts
ANSWER_EXTRACTION_PROMPT = """You are an AI assistant tasked with extracting the answer from a student's attempt at a problem. 

The user will provide you with their attempt at a problem in the following format:

# Attempt
{attempt}

Your job is to extract the answer from the attempt. Typically the answer will be boxed.


Output the following in JSON format:
{
    "extracted_answer": <the extracted answer>
}"""


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

OPENAI_CLIENT = AsyncAzureOpenAI(
    api_version="2025-01-01-preview",
    azure_endpoint="https://jonathan-research.openai.azure.com",
)
