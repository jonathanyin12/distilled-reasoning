import json
import os
import re

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

load_dotenv()

client = AsyncAzureOpenAI(
    api_version="2025-01-01-preview",
    azure_endpoint="https://jonathan-dev.openai.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)


alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\\s|She\\s|It\\s|They\\s|Their\\s|Our\\s|We\\s|But\\s|However\\s|That\\s|This\\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r"\.{2,}"


# https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences, ensuring that sentence-ending punctuation inside quotations results in multiple sentences, each enclosed in quotations.

    SPLITTING RULES:
    We split the text into sentences for the following:
    - periods
    - question marks
    - exclamation marks
    - semicolons
    - newlines
    - some ellipses
    Note: we preserve symbols after the punctuation e.g. ."

    For ellipses, we split the text into sentences if the next non-whitespace character after it is a capital letter or a quotation mark

    If the text contains substrings "<prd>" or "<stop>", they would lead to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = re.sub(r"(?m)^( *\d+)[.] ", r"\1<listprd> ", text)

    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    text = re.sub(multiple_dots, "<ellipsis>", text)
    text = re.sub(
        r"<ellipsis>\s*([\"A-Z])", r"<ellipsis><stop>\1", text
    )  # if the next nonspace character after <ellipsis> is a capital letter or a quotation mark, split the sentence
    if "Ph.D." in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(
        alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        text,
    )
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)

    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    # <stop> is added one character after sentence-splitting punctuation if the next char is not alphanumeric (helps preserve any important symbols after the punctuation, i.e. .*  or ." or .')
    text = re.sub(r"<stop>(\W)", r"\1<stop>", text)
    text = text.replace(";", ";<stop>")
    text = text.replace("\n", "<stop>")

    text = text.replace("<prd>", ".")
    text = text.replace("<ellipsis>", "...")
    text = text.replace("<listprd>", ".")

    sentences = text.split("<stop>")
    cleaned_sentences = []
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if len(sentence) == 0:
            continue
        # if the sentence is a single character, add it to the previous sentence unless it's the first sentence, in which case add it to the next sentence
        if len(cleaned_sentences) > 0:
            if i == 1 and len(cleaned_sentences[0]) == 1:
                # if first sentence is a single character, add next sentence to it
                cleaned_sentences[0] = cleaned_sentences[0] + sentence
            elif i > 0 and len(sentence) == 1:
                # if current sentence is a single character, add it to the previous sentence
                cleaned_sentences[-1] = cleaned_sentences[-1] + sentence
            else:
                cleaned_sentences.append(sentence)
        else:
            cleaned_sentences.append(sentence)
    return cleaned_sentences


def get_context(text, sentence, context_length=1000):
    sentence_index = text.find(sentence)
    start_index = max(0, sentence_index - context_length)
    end_index = min(len(text), sentence_index + len(sentence))
    return "... " + text[start_index:end_index] + " ..."


async def describe_first_correct_solution_location(question, answer, reasoning):
    prompt = f"""You are an AI assistant tasked with determining when the student reaches the correct answer in their reasoning process for the first time.

Here is the problem:
{question}

Here is the correct answer:
{answer}

Here is the student's full reasoning process:
{reasoning}


You goal is to describe where in the student's reasoning process they first reached the correct answer.
- They likely checked their work after they first reached the correct answer. However, only consider the very first time the correct answer is proposed as the answer to the question(i.e. before they started checking their work).

Provide direct quotes from the reasoning process to help describe where the correct answer first appears.


Output a JSON object with the following field:
{{  
    "response": <where in the reasoning process the correct answer first appears>,
}}"""

    messages = [
        {"role": "user", "content": prompt},
    ]

    response = await client.chat.completions.create(
        model="o3-mini",
        reasoning_effort="high",
        messages=messages,
        response_format={"type": "json_object"},
    )
    output_data = json.loads(response.choices[0].message.content)
    return output_data["response"]


async def choose_first_correct_solution(
    question, answer, reasoning, location, excerpts
):
    formatted_excerpts = "\n\n".join(
        [f"{i + 1}. {excerpt}" for i, excerpt in enumerate(excerpts)]
    )
    prompt = f"""You are an AI assistant tasked with finding the first time the student reaches the correct answer in their reasoning process.

Here is the problem:
{question}

Here is the correct answer:
{answer}


Here is the student's full reasoning process:
{reasoning}


Below, you are given a number of excerpts taken from the student's full reasoning process. They are ordered by when they appear in the student's reasoning process. One of the excerpts contains the first time the student reaches the correct answer. Your job is to determine which excerpt it is. 

Here are the excepts:
{formatted_excerpts}



Here is a description of the location of the first correct answer:
{location}

Reference the description to determine which excerpt contains the first correct answer. The correct excerpt contains the correct answer in the last sentence of the excerpt.


Output a json object with the following field:
{{
    "excerpt_index": <1-indexed index of the excerpt that contains the first correct solution>
}}
"""
    messages = [
        {"role": "user", "content": prompt},
    ]
    response = await client.chat.completions.create(
        model="o3-mini",
        reasoning_effort="high",
        messages=messages,
        response_format={"type": "json_object"},
    )
    cost = (
        response.usage.prompt_tokens * 1.10 / 1e6
        + response.usage.completion_tokens * 4.40 / 1e6
    )
    output_data = json.loads(response.choices[0].message.content)
    index = int(output_data["excerpt_index"]) - 1

    return index


async def choose_first_correct_solution_sentence(
    question, answer, reasoning, excerpt, sentences
):
    formatted_sentences = "\n\n".join(
        [f"{i + 1}. {sentence}" for i, sentence in enumerate(sentences)]
    )
    prompt = f"""You are an AI assistant tasked with determining when a student reached the correct answer in their reasoning process.

Here is the problem:
{question}

Here is the correct answer:
{answer}

Here is the student's full reasoning process:
{reasoning}


So far, you've determined that the first correct answer is reached somewhere in the following excerpt:
{excerpt}


Below, you are given a list of sentences from the excerpt. One of the sentences contains the first time the student reaches the correct answer. Your job is to determine which sentence it is. Here are the sentences:

{formatted_sentences}


The student must explicitly propose the correct answer for it to be considered a correct answer.
- Some sentences may contain the value of the answer but not be proposing it as the answer i.e. it is used in some other context.


Output a json object with the following fields:
{{
    "reasoning": <Reasoning about which the sentence is the first correct answer>,
    "sentence_index": <1-indexed index of the sentence, must be between 1 and {len(sentences)}>
}}
"""
    messages = [
        {"role": "system", "content": prompt},
    ]
    response = await client.chat.completions.create(
        model="o3-mini",
        reasoning_effort="high",
        messages=messages,
        response_format={"type": "json_object"},
    )
    cost = (
        response.usage.prompt_tokens * 1.10 / 1e6
        + response.usage.completion_tokens * 4.40 / 1e6
    )
    output_data = json.loads(response.choices[0].message.content)
    index = int(output_data["sentence_index"]) - 1
    return index
