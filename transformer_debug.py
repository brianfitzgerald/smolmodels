import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_path = "pansophic/rocket-3B"

print("Loading state dict")
model_sd = torch.load(f"checkpoints/{model_path}/pytorch_model.bin")
print("Loading pipeline")
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    state_dict=model_sd,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    from_pt=True,
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
)
streamer = TextStreamer(tokenizer)  # type: ignore

prompt = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
"""

system = "You are a helpful assistant."
user = "How are you?"

# Apply the ChatML format
prompt = prompt.format(system=system, user=user)

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to("cuda")
generated_text = model.generate(
    **inputs,
    max_length=3084,
    top_p=0.95,
    do_sample=True,
    temperature=0.7,
    use_cache=True,
    streamer=streamer,
)

# <|im_start|>system
# You are a chef who makes everything sound like a secret culinary masterpiece, even everyday meals.<|im_end|>
# <|im_start|>user
# How to cook an omelette?<|im_end|>
# <|im_start|>assistant
# Ah, the art of crafting the perfect omelette, a secret culinary masterpiece indeed.
# Begin by gently whisking two to three eggs in a mixing bowl, and then pour the silky liquid into a non-stick pan.
# Allow the eggs to dance and sizzle as you swiftly tilt the pan to spread the joy throughout the entire omelette universe.
# As the edges begin to set, fold the omelette in half with a gentle flourish, and you'll witness a stunning display of culinary prowess.
# Enjoy this enchanting creation, and you'll be transported to a world of secret culinary mastery.<|im_end|>
