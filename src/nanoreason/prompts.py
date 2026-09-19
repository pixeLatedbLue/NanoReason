from __future__ import annotations

STRATEGYQA_SYSTEM = "Answer each question with only 'yes' or 'no'. No explanation."
GSM8K_SYSTEM = "Solve the math problem step by step. End your answer with: #### <number>"
MMLU_SYSTEM = "Answer the multiple-choice question with only one letter: A, B, C, or D."
AQUA_SYSTEM = "Solve the problem step by step, then end with: #### <letter> (A, B, C, D, or E)."

STRATEGYQA_FEW_SHOT = [
    {"q": "Did Aristotle live before the printing press was invented?", "a": "yes"},
    {"q": "Is the Great Wall of China visible from space with the naked eye?", "a": "no"},
    {"q": "Can a human survive without any sleep for a month?", "a": "no"},
]

GSM8K_FEW_SHOT = [
    {
        "q": "Jane has 5 apples. She buys 7 more. How many apples does Jane have?",
        "a": "Jane starts with 5 apples and buys 7 more.\n5 + 7 = 12\n#### 12",
    },
    {
        "q": "A train travels at 60 km/h for 3 hours. How far did it go?",
        "a": "Distance = speed * time = 60 * 3 = 180 km.\n#### 180",
    },
]


def chat_prompt(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def strategyqa_prompt(tokenizer, question: str, few_shot: bool = True) -> str:
    messages = [{"role": "system", "content": STRATEGYQA_SYSTEM}]
    if few_shot:
        for ex in STRATEGYQA_FEW_SHOT:
            messages.extend(
                [
                    {"role": "user", "content": ex["q"]},
                    {"role": "assistant", "content": ex["a"]},
                ]
            )
    messages.append({"role": "user", "content": question})
    return chat_prompt(tokenizer, messages)


def gsm8k_prompt(tokenizer, question: str, few_shot: bool = True) -> str:
    messages = [{"role": "system", "content": GSM8K_SYSTEM}]
    if few_shot:
        for ex in GSM8K_FEW_SHOT:
            messages.extend(
                [
                    {"role": "user", "content": ex["q"]},
                    {"role": "assistant", "content": ex["a"]},
                ]
            )
    messages.append({"role": "user", "content": question})
    return chat_prompt(tokenizer, messages)


AQUA_FEW_SHOT = [
    {
        "q": "A shop sells a pen for $3. How much for 4 pens?\nOptions:\nA)9\nB)12\nC)15\nD)7\nE)6",
        "a": "Each pen costs $3.\n3 * 4 = 12\n#### B",
    },
]


def aqua_prompt(tokenizer, question: str, few_shot: bool = True) -> str:
    messages = [{"role": "system", "content": AQUA_SYSTEM}]
    if few_shot:
        for ex in AQUA_FEW_SHOT:
            messages.extend(
                [
                    {"role": "user", "content": ex["q"]},
                    {"role": "assistant", "content": ex["a"]},
                ]
            )
    messages.append({"role": "user", "content": question})
    return chat_prompt(tokenizer, messages)


def mmlu_prompt(tokenizer, sample: dict) -> str:
    choices = sample["choices"]
    user = (
        f"{sample['question']}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}"
    )
    return chat_prompt(
        tokenizer,
        [{"role": "system", "content": MMLU_SYSTEM}, {"role": "user", "content": user}],
    )
