import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import pandas as pd

# Define the device (use "cuda" if available, otherwise use "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load the Kaggle dataset (replace with your dataset)
data = pd.read_csv("legal_docs.csv")

# Replace 'actual_text_column_name' with the correct column name containing the text prompts
actual_text_column_name = "clause_text"
data[actual_text_column_name] = data[actual_text_column_name].fillna("")  # Fill missing values with an empty string

texts = data[actual_text_column_name].tolist()
encoded_texts = [tokenizer.encode(text, add_special_tokens=True) for text in texts]

# Prepare the training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
)

# Create a PyTorch Dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_texts):
        self.encoded_texts = encoded_texts

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded_texts[idx])

train_dataset = TextDataset(encoded_texts)

# Define the training function
def model_training_function():
    return model.to(device)

# Initialize the Trainer
trainer = Trainer(
    model=model_training_function,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()
