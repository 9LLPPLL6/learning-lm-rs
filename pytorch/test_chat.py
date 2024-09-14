from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_dir = "models/chat"
torch.set_printoptions(threshold=float("inf"))
model = AutoModelForCausalLM.from_pretrained(model_dir)

tokenizer = AutoTokenizer.from_pretrained(model_dir)

message = "<|im_start|>system\nYou are a highly knowledgeable and friendly assistant. Your goal is to understand and respond to user inquiries with clarity. Your interactions are always respectful helpful, and focused on delivering the most accurate information to the user.<|im_end|>\n<|im_start|>user\nHey! Got a question for you!<|im_end|>\n<|im_start|>assistant\nSure! What's it?<|im_end|>\n<|im_start|>user\nWhat are some potential applications for quantum computing?<|im_end|>\n<|im_start|>assistant"

input = tokenizer(message, return_tensors="pt")
output_dict = {}

def hook_fn(layer_name):
    def hook(module, input, output):
        output_dict[layer_name] = {
            "input": input,
            "output": output
        }
    return hook

for name, layer in model.named_modules():
    # layer_name = f"transformer_layer_{name}"
    # print(layer_name)
    layer.register_forward_hook(hook_fn(name))
    
with torch.no_grad():
    model(**input)
    
# self_out_tensor = output_dict['model.layers.0.self_attn']['output'][0]
# self_int_tensor = output_dict["model.layers.0.self_attn"]['input']

# # print("self input tensor shape: {}", self_int_tensor.shape)
# print(self_int_tensor)
# print("self output tensor shape: {}", self_out_tensor.shape)
# print(self_out_tensor)