
import sys
import os

import argparse
import torch
import torch._dynamo
import math
import torch.nn.functional as F
from einops import rearrange
from util.utils import set_dtype, seqlen_to_mask, torch_cuda_identify, time_stamp_cudasync, transpose_for_scores
from util.masks import generate_causal_mask

import warnings
from torch.jit import TracerWarning
warnings.filterwarnings("ignore", category=TracerWarning)

import torch._dynamo
torch._dynamo.config.suppress_errors = True


def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_q = Q.size(2)
        seq_kv = K.size(2)
        mask = torch.tril(torch.ones(seq_q, seq_kv, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum('bhqk,bhkd->bhqd', attention_weights, V)
    return output


def new_gelu(input):
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


# Bert-small | Bert-base | Bert-large ----------------------------------------
# ----------------------------------------------------------------------------
def bert_fwd_std(mask):
    with torch.no_grad():
        hidden_states = input_from_tensor
        for layer in range(layer_num):
            input_tensor = hidden_states

            qkv = qkv_kernel[layer]  + qkv_bias[layer]
            q1, k1, v1 = qkv.chunk(3, dim=-1)
            q1 = transpose_for_scores(q1, head_num, head_size)  
            k1 = transpose_for_scores(k1, head_num, head_size)  
            v1 = transpose_for_scores(v1, head_num, head_size)
            
            q = q1.permute(0, 2, 1, 3).contiguous()
            k = k1.permute(0, 2, 1, 3).contiguous()
            v = v1.permute(0, 2, 1, 3).contiguous() 
            q = rearrange(q, 'b t h d -> (b h) t d')
            k = rearrange(k, 'b s h d -> (b h) d s')
            softmax_scale = 1.0 / math.sqrt(head_size)
            
            scores = torch.empty(batch_size * head_num, seq_len, seq_len, dtype=q.dtype, device=q.device)
            scores = rearrange(torch.baddbmm(scores, q, k, beta=1.0, alpha=softmax_scale),
                               '(b h) t s -> b h t s', h=head_num)
            causal_mask = torch.triu(torch.full((seq_len, seq_len), -10000.0, device=scores.device), 1)
            scores = scores + causal_mask.to(dtype=scores.dtype)
            
            attention = torch.softmax(scores, dim=-1)
            attention_drop = F.dropout(attention, dropout_p)
            h = torch.einsum('bhts,bshd->bthd', attention_drop , v).to(dtype=q.dtype)

            new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
            hidden_states = h.reshape(new_context_layer_shape)
                    
            # ------------------------------------------------------------ Attention End
            hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
            hidden_states = hidden_states + input_tensor
            hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
                                        weight=attr_output_layernorm_gamma[layer], bias=attr_output_layernorm_beta[layer])
            residual = hidden_states
        
            hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer] 
            hidden_states = F.gelu(hidden_states) 
            hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]
            hidden_states = hidden_states + residual 
            hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),  
                                        weight=output_layernorm_gamma[layer], bias=output_layernorm_beta[layer])  
        
            transformer_output[layer] = hidden_states


from transformers.cache_utils import DynamicCache
from transformers.activations import ACT2FN

def llama3_base_fwd_std(mask):
    hidden_states = input_from_tensor

    past_key_values = DynamicCache()
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    cache_position: torch.Tensor = torch.arange(past_seen_tokens, past_seen_tokens + input_from_tensor.shape[1], device=input_from_tensor.device)
    position_ids = cache_position.unsqueeze(0)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_size, 2, dtype=torch.int64).to(device=hidden_states.device, dtype=torch.float) / head_num))
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(hidden_states.device)
    position_ids_expanded = position_ids[:, None, :].float()
    device_type = hidden_states.device.type if isinstance(hidden_states.device.type, str) and hidden_states.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * 1.0
        sin = emb.sin() * 1.0
    position_embeddings = (cos.to(dtype=hidden_states.dtype), sin.to(dtype=hidden_states.dtype))
    

    for layer in range(layer_num):
        residual = hidden_states
        
        hidden_size = hidden_states.shape[-1]
        weight = torch.ones(hidden_size, device=hidden_states.device)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + 1e-6)
        hidden_statea = weight * hidden_states.to(input_dtype)

        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_num)
    
        qkv = qkv_kernel[layer]  + qkv_bias[layer]
        q, k, v = qkv.chunk(3, dim=-1)
        q1 = transpose_for_scores(q, head_num, head_size)  
        k1 = transpose_for_scores(k, head_num, head_size)  
        v1 = transpose_for_scores(v, head_num, head_size)
            
        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q1 = (q1 * cos) + (torch.cat((-q1[..., q1.shape[-1] // 2 :], q1[..., : q1.shape[-1] // 2]), dim = -1) * sin)
        k1 = (k1 * cos) + (torch.cat((-k1[..., k1.shape[-1] // 2 :], k1[..., : k1.shape[-1] // 2]), dim = -1) * sin)

        q = q1.permute(0, 2, 1, 3)
        k = k1.permute(0, 2, 1, 3)
        v = v1.permute(0, 2, 1, 3)
        q = rearrange(q, 'b t h d -> (b h) t d')
        k = rearrange(k, 'b s h d -> (b h) d s')
        softmax_scale = 1.0 / math.sqrt(head_size)
        
        scores = torch.empty(batch_size * head_num, seq_len, seq_len, dtype=q.dtype, device=q.device)
        scores = rearrange(torch.baddbmm(scores, q, k, beta=1.0, alpha=softmax_scale),
                           '(b h) t s -> b h t s', h=head_num)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), -10000.0, device=scores.device), 1)
        scores = scores + causal_mask.to(dtype=scores.dtype)
        
        attention = torch.softmax(scores, dim=-1)
        attention_drop = F.dropout(attention, dropout_p)
        h = torch.einsum('bhts,bshd->bthd', attention_drop , v).to(dtype=q.dtype)

        new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
        hidden_states = h.reshape(new_context_layer_shape)
                
        # ------------------------------------------------------------ Attention End

        hidden_states = residual + hidden_states
        residual = hidden_states

        hidden_size = hidden_states.shape[-1]
        weight = torch.ones(hidden_size, device=hidden_states.device)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + 1e-6)
        hidden_states = weight *hidden_states.to(input_dtype)

        act_fn = ACT2FN["silu"]
        hidden_states = act_fn(hidden_states) * hidden_states
        hidden_states = residual + hidden_states

        transformer_output[layer] = hidden_states
        
if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    device = torch_cuda_identify(print_info = False)
    torch._dynamo.config.cache_size_limit = 64
    
    is_4080_laptop = False
    is_4090 = False
    is_A100 = False
    gpu_name = torch.cuda.get_device_name()
    if "NVIDIA GeForce RTX 4080 Laptop GPU" in gpu_name:
        is_4080_laptop = True
    if "NVIDIA GeForce RTX 4090" in gpu_name:
        is_4090 = True
    if "NVIDIA A100-PCIE-40GB" in gpu_name:
        is_A100 = True
    
    parser = argparse.ArgumentParser(description="Benchmark BERT/LLaMA with torch attention and causal mask")
    parser.add_argument('--method', type=str, default="TorchNative", help='TorchNative or TorchCompile')
    parser.add_argument('--model', type=str, default="bert_base", help='Sequence length (default: 1)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length (default: 256)')
    args = parser.parse_args() 

    model_selection  = args.model
    method_selection = args.method
    
    head_size = 64
    seq_len   = args.seq_len
    batch_size = args.batch_size
    
    data_type  = torch.float16
    dtype = "fp16"
    running_device = "cuda"

    data_type  = torch.float16
    running_device = "cuda"
    dropout_p = 0.0
    layer_num=1
    
    warmup_iters = 10
    running_iters = 10
    if model_selection == "bert_small":
        inference_model=bert_fwd_std
        head_num=8
        layer_num=6
        head_size = 64
    elif model_selection == "bert_base":
        inference_model=bert_fwd_std
        head_num=12
        layer_num=12
        head_size = 64
    elif model_selection == "bert_large":
        inference_model=bert_fwd_std   
        head_num=16
        layer_num=24
        head_size = 64   
    elif model_selection == "llama_small":
        inference_model = llama3_base_fwd_std
        head_num = 8
        layer_num=16
        head_size = 64
    elif model_selection == "llama_base":
        inference_model = llama3_base_fwd_std
        head_num = 16
        layer_num=24
        head_size = 64
    else:
        raise ValueError(f"Unsupported model: {model_selection}")
    hidden_dim = head_num * head_size 

    if method_selection == "TorchNative":
        run_compiled = False
    elif(method_selection == "TorchCompile"):
        run_compiled = True
        if(is_4080_laptop): 
            run_compiled = False
    else:
        raise ValueError(f"Unsupported method: {method_selection}")

    
    avg_seq_len = seq_len 
    low, high = (2 * avg_seq_len - seq_len, seq_len + 1)
    input_lens = torch.randint(low=low, high=high, size=(batch_size,))
    seqlen_mask = seqlen_to_mask(input_lens, seq_len)
    attr_mask   = set_dtype(torch.tile(seqlen_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len).cuda(), "fp16")
    
    is_causal = True
    mask = generate_causal_mask(attr_mask).cuda()
        
    qkv_kernel_raw              = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_kernel          = [set_dtype(torch.zeros(hidden_dim, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_bias            = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_layernorm_gamma = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_layernorm_beta  = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    inter_kernel                = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    inter_bias                  = [set_dtype(torch.zeros(hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_kernel               = [set_dtype(torch.zeros(hidden_dim * 4, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_bias                 = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_layernorm_gamma      = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_layernorm_beta       = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    
    input_from_tensor           = set_dtype(torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)
    qkv_bias                    = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    qkv_kernel                  = [set_dtype(torch.zeros(batch_size, seq_len, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
   
   
    transformer_output = [None for _ in range(layer_num)]
    runner = inference_model
    label = "Torch Native"
    if run_compiled:
        runner = torch.compile(inference_model, mode='default', backend='inductor')
        label = "Torch Compile"

    for i in range(warmup_iters + running_iters):
        if i == warmup_iters:    
            t_start = time_stamp_cudasync()
        runner(mask)
        output = transformer_output[-1]
        
    t_end = time_stamp_cudasync()
    elapsed_time = (t_end - t_start) * 1000 / running_iters
    print("e2e {} | bs:{} | seq:{}  |  {} : {:.3f} ms / iter".format(model_selection, batch_size, args.seq_len, label, elapsed_time)) 



        
        
        
        
        
