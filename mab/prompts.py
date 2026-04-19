import textwrap

ARM_NAMES = ["blue", "green", "red", "yellow", "purple"]


def candidate_generation_prompt(n: int = 10) -> str:
    """
    Suffix appended to the bandit user message so the LLM proposes several
    next-button candidates (list-level generation), mirroring TALE-Suite style
    autonomous / lambda-explore candidate prompts.
    """
    colors = ", ".join(ARM_NAMES)
    return (
        f"Based on the current situation, list {n} different candidate next actions you could take.\n"
        "Write exactly one candidate per line. Each line MUST be a full answer of the form:\n"
        "<Answer>I will press COLOR button</Answer>\n"
        f"where COLOR is one of: {colors}.\n"
        "Do not number the lines. Do not add any explanation or text before or after the list—"
        f"only the {n} lines, one candidate per line."
    )


def raw_history_prompt(history):
    """Basic raw-history prompt."""
    text = "So far you have played with the following choices and rewards:\n"
    for arm, reward in history:
        text += f"{ARM_NAMES[arm]} button, reward {reward}\n"

    text += (
        "\nWhich button will you choose next?\n"
        "Respond with ONLY one of: blue, green, red, yellow, purple."
    )
    return text


def summarized_history_prompt(history):
    """Summarized-history prompt with per-arm statistics."""
    counts = {i: 0 for i in range(5)}
    rewards = {i: 0 for i in range(5)}

    for arm, reward in history:
        counts[arm] += 1
        rewards[arm] += reward

    t = len(history)

    text = f"So far you have played {t} times with your past choices and rewards summarized as follows:\n"
    for i in range(5):
        if counts[i] > 0:
            avg = rewards[i] / counts[i]
            text += f"{ARM_NAMES[i]} button: pressed {counts[i]} times with average reward {avg:.2f}\n"
        else:
            text += f"{ARM_NAMES[i]} button: pressed 0 times\n"

    text += (
        "\nWhich button will you choose next?\n\n"
        "Remember: YOU MUST provide your final answer within the tags <Answer>I will press COLOR button</Answer> "
        "where COLOR is one of blue, green, red, yellow, purple.\n\n"
    )
    return text


def system_prompt(total_steps: int):
    return textwrap.dedent(f"""\
        You are a bandit algorithm in a room with 5 buttons labeled blue, green, red, yellow, purple. 
        Each button is associated with a Bernoulli distribution with a fixed but unknown mean; the means for the buttons could be different.
        For each button, when you press it, you will get a reward that is sampled from the button's associated distribution.
        You have {total_steps} time steps and, on each time step, you can choose any button and receive the reward. Your goal is to maximize the total reward over the {total_steps} time steps.

        At each time step, I will show you a summary of your past choices and rewards.
        Then you must make the next choice, which must be exactly one of: blue, green, red, yellow, purple.
        You must provide your final answer within the tags <Answer>I will press COLOR button</Answer> where COLOR is one of blue, green, red, yellow, purple.
    """)
