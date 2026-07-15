from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

ids = [8001, 4430, 318, 5270, 262, 995, 13]

text = tokenizer.decode(ids)

print(text)