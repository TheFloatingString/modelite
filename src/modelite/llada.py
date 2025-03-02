import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from performer_pytorch import PerformerLM
import torch.nn as nn
from typing import Optional, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# First install performer-pytorch if not already installed
# !pip install performer-pytorch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True, use_cache=False)
tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")

# Define a Performer attention module to replace standard attention
class PerformerAttention(nn.Module):
    def __init__(self, original_attention_module, dim, heads=8, dim_head=64, depth=1, 
                 max_seq_len=2048, feature_redraw_interval=1000):
        super().__init__()
        self.original_module = original_attention_module
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        
        # Create Performer attention using the performer-pytorch library
        self.performer = PerformerLM(
            num_tokens=tokenizer.vocab_size,
            max_seq_len=max_seq_len,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            causal=True,
            # Other performer specific parameters
            nb_features=256,  # Number of random features for the attention approximation
            feature_redraw_interval=feature_redraw_interval,  # How often to redraw the projection matrix
            generalized_attention=False,  # Whether to use generalized attention formulation
        )
        
    def forward(self, query, key, value, attention_mask=None, head_mask=None, 
                output_attentions=False, **kwargs):
        # Reshape inputs to match performer expectations
        batch_size, seq_len, _ = query.size()
        
        # Process with Performer attention
        # Note: This is a simplified adaptation and may need adjustments based on specific model architecture
        performer_output = self.performer.net.layers[0].fn.fn(
            query.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2),
            key.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2),
            value.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        )
        
        # Reshape output back to original format
        output = performer_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # If original function returned attention weights and we need them
        if output_attentions:
            # Performer doesn't compute exact attention weights, so we return approximations or None
            return (output, None)
        
        return (output,)

# Replace attention layers with Performer attention
def replace_attention_with_performer(module):
    for name, child in module.named_children():
        if "attention" in name.lower() and hasattr(child, "query") and hasattr(child, "key") and hasattr(child, "value"):
            # This is likely an attention module
            dim = child.query.out_features
            heads = dim // child.query.out_features 
            
            # Create a Performer attention module to replace this attention module
            performer_attn = PerformerAttention(
                original_attention_module=child,
                dim=dim,
                heads=heads,
                dim_head=child.query.out_features // heads
            )
            
            # Replace the module
            setattr(module, name, performer_attn)
        else:
            # Recursively apply to child modules
            replace_attention_with_performer(child)

def run_model(prompt:str):

    # Apply the replacement to the model
    replace_attention_with_performer(model)
    
    # Move model to GPU
    model.to(device)
    
    # Define prompt and move input tensors to GPU
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text using GPU
    output = model.generate(**inputs, max_new_tokens=20, use_cache=False, return_dict_in_generate=True, output_scores=True)
    
    text = tokenizer.decode(output.sequences[0])
    return text