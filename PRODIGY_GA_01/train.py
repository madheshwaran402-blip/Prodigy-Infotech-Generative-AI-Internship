from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load dataset
dataset = load_dataset(
    "text",
    data_files={"train": "dataset/data.txt"}
)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)

print("Starting Training...\n")

trainer.train()

print("\nTraining Completed!")

model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("Model Saved Successfully!")