# Description: This file contains the function to create a prompt for the RealTime task.

def create_prompt(question, options, GT):
    GT_index = ord(GT) - ord('A')
    correct_answer = options[GT_index]
    options_text = "\n".join([f"{chr(65 + i)}: {option}" for i, option in enumerate(options)])
    PROMPT_TEMPLATE = f"""
Question: {question}
Options: 
{options_text}
Answer with the option's letter from the given choices directly.
"""
    correct_option = chr(65 + options.index(correct_answer))
    return PROMPT_TEMPLATE, correct_option

# Example
if __name__ == "__main__":
    question = "What is the capital of France?"
    options = ["Paris", "London", "Berlin", "Madrid"]
    GT = "A"
    prompt, correct_option = create_prompt(question, options, GT)
    print(prompt)
    print(f"Correct option: {correct_option}")


"""
Question: What is the capital of France?
Options: 
A: Paris
B: London
C: Berlin
D: Madrid
Answer with the option's letter from the given choices directly.

Correct option: A
"""