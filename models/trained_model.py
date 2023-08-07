import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import nbformat as nbf
import os

class CodeGenerationBart:
    def __init__(self):
        self.prompts_df = pd.read_csv('data/prompts.csv')
        self.model = None
        self.tokenizer = None

    def train_model(self):
        print("Training a new model...")
        
        # Load the pre-trained BART model and tokenizer
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

        # Define a custom Dataset for code generation
        class CodeGenerationDataset(Dataset):
            def __init__(self, prompts_df, tokenizer):
                self.prompts = prompts_df['prompt']
                self.codes = prompts_df['code_snippet']
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.prompts)

            def __getitem__(self, idx):
                source_text = self.prompts[idx]
                target_text = self.codes[idx]
                source_ids = self.tokenizer.encode(source_text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
                target_ids = self.tokenizer.encode(target_text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
                return {
                    'input_ids': source_ids.squeeze(),
                    'labels': target_ids.squeeze()
                }

        # Create an instance of the custom dataset
        train_dataset = CodeGenerationDataset(self.prompts_df, self.tokenizer)

        # Define training arguments for the Seq2SeqTrainer
        training_args = Seq2SeqTrainingArguments(
            output_dir='./data/models',
            overwrite_output_dir=True,
            per_device_train_batch_size=4,
            save_steps=1000,
            save_total_limit=2,
            num_train_epochs=5,
            predict_with_generate=True,
            evaluation_strategy="steps",
            eval_steps=1000,
            logging_steps=100,
            learning_rate=1e-4,
        )

        # Create a Seq2SeqTrainer instance for training
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Train the model
        trainer.train()
        print("Model training completed.")

        # Save the model
        self.model.save_pretrained('./data/models/bart-large-code-generation')
        print("Model saved to ./data/models/bart-large-code-generation")


    def test_model(self):
        print("Testing the existing model...")

        # Check if the model exists
        model_path = './data/models/bart-large-code-generation'
        if not os.path.exists(model_path):
            print("Model does not exist. Training a new model...")
            self.train_model()
            return

        # Load the model
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

        test_prompts = [
            "Calculate the average age of passengers.",
            "Find the total number of passengers.",
            "Calculate the percentage of passengers who survived."
        ]

        nb = nbf.v4.new_notebook()

        self.model.eval()
        for idx, prompt in enumerate(test_prompts, start=1):
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            with torch.no_grad():
                output = self.model.generate(input_ids, max_length=128, num_beams=5, no_repeat_ngram_size=2, top_p=0.9, top_k=50)
            predicted_code = self.tokenizer.decode(output[0], skip_special_tokens=True)

            code_cell = nbf.v4.new_code_cell()
            code_cell.source = f"# Prompt {idx}: {prompt}\n\n{predicted_code}"
            nb.cells.append(code_cell)

        notebook_path = 'output/bart/output.ipynb'
        with open(notebook_path, 'w') as f:
            nbf.write(nb, f)

        print(f"Notebook saved to {notebook_path}")
        print("Model testing completed.")

    def run(self):
        print("Welcome to the BART Code Generation System!")
        choice = input("Do you want to train a new model and then test it? (Y/N): ").strip().lower()

        if choice == "y":
            self.train_model()
            self.test_model()
        elif choice == "n":
            self.test_model()
        else:
            print("Invalid choice. Please enter 'Y' or 'N'.")
