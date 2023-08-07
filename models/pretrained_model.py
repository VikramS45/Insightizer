import pandas as pd
import openai
import nbformat as nbf

class CodeGenerationOpenAI:
    def __init__(self):
        self.test_prompts = [
            "Calculate the average age of passengers.",
            "Find the total number of passengers.",
            "Calculate the percentage of passengers who survived."
        ]

    def generate_code(self, prompt, key):
        openai.api_key = key
        response = openai.Completion.create(
            engine="text-davinci-003",  # Choose an appropriate engine
            prompt=prompt + "\nCode:",
            max_tokens=100  # Adjust as needed
        )
        return response.choices[0].text.strip()

    def run(self, key):
        nb = nbf.v4.new_notebook()

        for idx, prompt in enumerate(self.test_prompts, start=1):
            predicted_code = self.generate_code(prompt, key)

            code_cell = nbf.v4.new_code_cell()
            code_cell.source = f"# Prompt {idx}: {prompt}\n\n{predicted_code}"
            nb.cells.append(code_cell)

        notebook_path = 'output/openai/output.ipynb'
        with open(notebook_path, 'w') as f:
            nbf.write(nb, f)

        print(f"Notebook saved to {notebook_path}")
