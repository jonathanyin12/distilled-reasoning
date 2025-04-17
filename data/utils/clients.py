import os

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI

load_dotenv()


# Client configuration
FIREWORKS_CLIENT = AsyncOpenAI(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    base_url="https://api.fireworks.ai/inference/v1",
)
XAI_CLIENT = AsyncOpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

CLIENTS = {
    "claude-3-7": AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
    "deepseek-r1": FIREWORKS_CLIENT,
    "deepseek-r1-distill-qwen-14b": FIREWORKS_CLIENT,
    "deepseek-r1-distill-qwen-7b": FIREWORKS_CLIENT,
    "deepseek-r1-distill-qwen-1.5b": FIREWORKS_CLIENT,
    "qwq-32b": FIREWORKS_CLIENT,
    "grok-3-mini-low": XAI_CLIENT,
    "grok-3-mini-high": XAI_CLIENT,
}

OPENAI_CLIENT = AsyncAzureOpenAI(
    api_version="2025-01-01-preview",
    azure_endpoint="https://jonathan-dev.openai.azure.com",
)
