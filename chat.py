"""Minimal chat loop:  python chat.py [path_to_this_folder]"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path = sys.argv[1] if len(sys.argv) > 1 else "."
tok = AutoTokenizer.from_pretrained(path)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(path).to(dev)
if dev == "cuda": model = model.half()
model.eval()
history = []
print("Type 'quit' to exit, 'reset' to clear history.")
while True:
    user = input("You: ").strip()
    if user == "quit": break
    if user == "reset": history = []; continue
    history.append({"role": "user", "content": user})
    ids = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(dev)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9,
                             repetition_penalty=1.1, pad_token_id=tok.pad_token_id)
    reply = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    history.append({"role": "assistant", "content": reply})
    print("Bot:", reply)
