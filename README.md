Insightzer - Empower, Explore, Excel: Unleash Data Insights

Confluence link - https://lv-knowledgemanagement.atlassian.net/l/cp/b77s1K7u

It is a proof-of-concept project that demonstrates the generation of Python code based on natural language prompts for performing basic data analytics. The project utilizes the Hugging Face Transformers library to train a language model and generate code snippets.

data: Contains dataset files, including titanic_dataset.csv for data analysis and prompts.csv for training prompts and corresponding code.
models: Stores trained model files.
config: Configuration files or settings if needed.
README.md: This file, provides an overview of the project, its structure, and usage instructions.
requirements.txt: Lists required Python packages and their versions.
.gitignore: Specifies files and directories to be ignored by version control.

Installation and Usage
Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # Use 'venv\Scripts\activate' on Windows

Install required packages:

pip install -r requirements.txt
Place your dataset (e.g., titanic_dataset.csv) in the data/ directory.

Prepare training data: Create a CSV file (prompts.csv) in the data/ directory with columns "prompt" and "code", containing training prompts and their corresponding Python code snippets.

Train the model (see code snippets) using the Hugging Face Transformers library. Store the trained model file (e.g., my_trained_model.pth) in the models/ directory.
"# GenAI_POC" 
