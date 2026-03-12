import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# -----------------------------
# Model Paths
# -----------------------------

BASE_MODEL_PATH = "models/base_model"
ADAPTER_PATH = "models/adapter"


# -----------------------------
# Load Tokenizer
# -----------------------------

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)


# -----------------------------
# Load Base Model
# -----------------------------

print("Loading base model (may take ~1 minute on CPU)...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="cpu"
)


# -----------------------------
# Load LoRA Adapter
# -----------------------------

print("Loading Kumaoni LoRA adapter...")

model = PeftModel.from_pretrained(model, ADAPTER_PATH)

model.eval()


print("\nKumaoni chatbot ready 🚀")
print("Type 'exit' to stop.\n")


# -----------------------------
# Chat Function
# -----------------------------

def chat(user_input):

    prompt = f"""### Instruction:
You are a Kumaoni assistant.
Always reply in Kumaoni language.
Use simple village-style Kumaoni sentences.

### Input:
{user_input}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            temperature=0.35,
            top_p=0.8,
            repetition_penalty=1.3,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the response part
    answer = response.split("### Response:")[-1].strip()

    print("\nBot:", answer, "\n")


# -----------------------------
# Interactive Loop
# -----------------------------

while True:

    user = input("You: ")

    if user.lower() in ["exit", "quit"]:
        print("Chat ended.")
        break

    chat(user)