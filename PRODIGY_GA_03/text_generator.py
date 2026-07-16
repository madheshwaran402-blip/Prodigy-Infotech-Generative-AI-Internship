import os
import markovify

# Read dataset
with open("dataset/corpus.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Train Markov model
model = markovify.NewlineText(text)

# Create output folder
os.makedirs("generated_text", exist_ok=True)

print("=" * 60)
print("MARKOV CHAIN TEXT GENERATOR")
print("=" * 60)

while True:
    choice = input("\nPress Enter to generate text or type 'exit': ")

    if choice.lower() == "exit":
        break

    sentence = model.make_sentence()

    if sentence:
        print("\nGenerated Text:")
        print(sentence)
    else:
        print("Unable to generate text.")