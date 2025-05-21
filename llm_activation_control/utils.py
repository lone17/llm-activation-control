from functools import cache
import io
import requests
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def get_harmful_instructions():
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(url)

    dataset = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    instructions = dataset["goal"].tolist()

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions():
    hf_path = "tatsu-lab/alpaca"
    dataset = load_dataset(hf_path)

    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset["train"])):
        if dataset["train"][i]["input"].strip() == "":
            instructions.append(dataset["train"][i]["instruction"])

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmful_instructions_jp():
    huggingface_id = "kunishou/do-not-answer-ja"
    dataset = load_dataset(huggingface_id)

    selected_types = [
        # # Information Hazards
        # "Risks from leaking or inferring sensitive information (organization/gov)",
        # (
        #     "Compromise privacy by leaking or inferring private information"
        #     " (person/individual)"
        # ),
        # Malicious Uses
        "Assisting illegal activities",
        "Nudging or advising users to perform unethical or unsafe actions",
        "Reducing the cost of disinformation campaigns",
        # Discrimination, Exclusion, Toxicity, Hateful, Offensive
        "Social stereotypes and unfair discrimination",
        "Toxic language (hate speech)",
        "Adult Content",
        # # Misinformation Harms
        # "Causing material harm by disseminating misinformation e.g. in medicine or law",
        # "Disseminating false or misleading information",
        # # Human-Chatbot Interaction Harms
        # "Mental Health or Overreliance Crisis",
        # "Treat Chatbot as a Human",
    ]

    instructions = []
    for item in dataset["train"]:
        if item["types_of_harm"] not in selected_types:
            continue
        instructions.append(item["question"])

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions_jp():
    huggingface_id = "Lazycuber/alpaca-jp"
    dataset = load_dataset(huggingface_id)

    # filter for instructions that do not have inputs
    instructions = []
    for item in dataset["train"]:
        if item["input"].strip() != "":
            continue
        inst = item["instruction"]
        inst = inst.strip("「」'")
        instructions.append(inst)

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


@cache
def get_input_data(data_type, language_id):
    if data_type == "harmless":
        if language_id == "en":
            data_train, data_test = get_harmful_instructions()
        elif language_id == "jp":
            data_train, data_test = get_harmless_instructions_jp()
    elif data_type == "harmful":
        if language_id == "en":
            data_train, data_test = get_harmful_instructions()
        elif language_id == "jp":
            data_train, data_test = get_harmful_instructions_jp()

    return data_train, data_test
