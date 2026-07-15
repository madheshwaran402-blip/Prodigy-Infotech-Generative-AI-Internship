from transformers import GPT2Tokenizer

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Read dataset
with open("dataset/data.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Tokenize
tokens = tokenizer(
    text,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

print("Original Text:\n")
print(text)

print("\nNumber of Tokens:")
print(len(tokens["input_ids"][0]))

print("\nToken IDs:")
print(tokens["input_ids"])