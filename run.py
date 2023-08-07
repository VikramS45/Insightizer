import os
import yaml
from models.pretrained_model import CodeGenerationOpenAI
from models.trained_model import CodeGenerationBart

class CodeGenerationApp:
    def __init__(self):
        self.config = self.load_config()
        self.api_key = self.config['openai']['api_key']
        self.bart = CodeGenerationBart()
        self.openai = CodeGenerationOpenAI()

    def load_config(self):
        config_path = os.path.join('config', 'config.yaml')
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        return config

    def main(self):
        print("\nWelcome to the Code Generation System!\n")
        choice = input("Choose a mode:\n1. Generate code using BART model\n2. Generate code using OpenAI's Codex\nEnter your choice (1 or 2): ")

        if choice == "1":
            self.bart.run()
        elif choice == "2":
            self.openai.run(self.api_key)
        else:
            print("Invalid choice. Please enter '1' or '2'.")

if __name__ == "__main__":
    app = CodeGenerationApp()
    app.main()
