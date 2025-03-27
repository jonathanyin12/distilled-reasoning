import json
from typing import Tuple
from .constants import CLIENTS, OPENAI_CLIENT, GRADING_PROMPT, MATCHING_PROMPT
from .response_processing import process_think_tagged_output, process_claude_response


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
        A tuple containing the attempt answer, correctness of the answer, and the explanation.
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
        reasoning_effort="high",
    )
    json_response = json.loads(response.choices[0].message.content)

    # Ensure attempt_answer is a string
    if isinstance(json_response["attempt_answer"], (int, float, bool)):
        attempt_answer = str(json_response["attempt_answer"])
    else:
        attempt_answer = json.dumps(json_response["attempt_answer"])

    correct = json_response["correct"]
    explanation = json_response["explanation"]

    return (
        attempt_answer,
        correct,
        explanation,
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
    question: str, question_type: str, solution: str, model: str, max_attempts: int = 16
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
            (
                attempt_answer,
                is_correct,
                explanation,
            ) = await verify_answer_correctness(question, attempt, solution)
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
