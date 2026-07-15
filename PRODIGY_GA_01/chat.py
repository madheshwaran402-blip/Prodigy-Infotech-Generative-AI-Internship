from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ================================
# Load Trained GPT-2 Model
# ================================
MODEL_PATH = "./model"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("\n===================================")
print("      Custom GPT-2 Chatbot")
print("===================================")
print("Type 'exit' to quit.\n")

# ================================
# Chat Loop
# ================================

while True:

    user_input = input("You : ")

    if user_input.lower() == "exit":
        print("\nThank you for using the chatbot!")
        break

    # Encode input
    inputs = tokenizer(
        user_input,
        return_tensors="pt"
    ).to(device)

    # Generate response
    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            # Number of generated tokens
            max_new_tokens=80,

            # Sampling
            do_sample=True,

            # Better creativity
            temperature=0.7,

            # Better word selection
            top_k=50,
            top_p=0.9,

            # Reduce repetition
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,

            # Stop properly
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only newly generated text
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    print("\nGPT-2 :", response)
    print()