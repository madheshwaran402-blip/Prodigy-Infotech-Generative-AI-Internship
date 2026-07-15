from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "Artificial Intelligence is changing the world."

tokens = tokenizer.tokenize(text)

token_ids = tokenizer.encode(text)

print("Original Text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)