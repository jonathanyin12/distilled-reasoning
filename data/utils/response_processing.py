from typing import Tuple


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
