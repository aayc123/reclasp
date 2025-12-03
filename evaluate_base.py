# import torch
# from modeling_llama import LlamaForCausalLM
# from transformers import AutoTokenizer
# from decoding import infer

# # 1. 加载模型
# print("Loading model...")
# model = LlamaForCausalLM.from_pretrained('/data/zn/model/models/Meta-Llama-3-8B', torch_dtype=torch.bfloat16)
# model = model.to('cuda:0').eval()
# tokenizer = AutoTokenizer.from_pretrained('/data/zn/model/models/Meta-Llama-3-8B')

# # 2. 准备测试样本
# xsum_example = '''The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed. Repair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water.
# Summary:'''

# # 3. 运行原始模型推理（不使用推测解码）
# print("Running base model inference...")
# result = infer(
#     model, 
#     tokenizer, 
#     xsum_example, 
#     generate_fn='base',  # 使用原始模型
#     max_new_tokens=512,
#     do_sample=False  # 贪心解码
# )

# # 4. 打印结果
# print(f"\n{'='*60}")
# print(f"Completion: {result['completion']}")
# print(f"{'='*60}")
# print(f"Time: {result['time']:.2f}s")
# print(f"Tokens generated: {result['generate_ids'].shape[1]}")


# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# print("Loading model with original Hugging Face implementation...")
# model = AutoModelForCausalLM.from_pretrained(
#     '/data/zn/model/models/Meta-Llama-3-8B', 
#     torch_dtype=torch.bfloat16,
#     device_map='cuda:0'
# )
# tokenizer = AutoTokenizer.from_pretrained('/data/zn/model/models/Meta-Llama-3-8B')

# # 测试样本
# prompt = '''The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed. Repair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water.
# Summary:'''

# print("\nGenerating with original HF model...")
# inputs = tokenizer(prompt, return_tensors='pt').to('cuda:0')

# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=100,
#         do_sample=False,
#         pad_token_id=tokenizer.eos_token_id
#     )

# completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(f"\n{'='*60}")
# print(f"Completion:\n{completion}")
# print(f"{'='*60}")


import torch
from modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer
import torch.nn.functional as F

print("Loading model...")
model = LlamaForCausalLM.from_pretrained('/data/zn/model/models/Meta-Llama-3-8B', torch_dtype=torch.bfloat16)
model = model.to('cuda:0').eval()
tokenizer = AutoTokenizer.from_pretrained('/data/zn/model/models/Meta-Llama-3-8B')

# 🔍 检查模型权重是否正确加载
print("\n=== Model Weight Check ===")
print(f"Model config: {model.config.num_hidden_layers} layers")
print(f"First layer attn weight mean: {model.model.layers[0].self_attn.q_proj.weight.mean().item():.6f}")
print(f"First layer attn weight std: {model.model.layers[0].self_attn.q_proj.weight.std().item():.6f}")
print(f"Last layer attn weight mean: {model.model.layers[-1].self_attn.q_proj.weight.mean().item():.6f}")
print(f"LM head weight mean: {model.lm_head.weight.mean().item():.6f}")
print(f"LM head weight std: {model.lm_head.weight.std().item():.6f}")

# 🔍 检查是否所有层都被正确初始化
print(f"\n=== Layer Initialization Check ===")
for i in [0, 15, 31]:
    layer = model.model.layers[i]
    print(f"Layer {i}:")
    print(f"  - self_attn.q_proj exists: {hasattr(layer.self_attn, 'q_proj')}")
    print(f"  - self_attn.k_proj exists: {hasattr(layer.self_attn, 'k_proj')}")
    print(f"  - self_attn.v_proj exists: {hasattr(layer.self_attn, 'v_proj')}")
    print(f"  - self_attn.o_proj exists: {hasattr(layer.self_attn, 'o_proj')}")
    print(f"  - self_attn.rotary_emb exists: {hasattr(layer.self_attn, 'rotary_emb')}")
    print(f"  - mlp exists: {hasattr(layer, 'mlp')}")

# 简单测试
prompt = "Hello, how are"
input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to('cuda:0')

print(f"\n=== Generation test ===")
generated = input_ids[0].tolist()
past_kv = None

for i in range(5):  # 只跑5步
    with torch.no_grad():
        if past_kv is None:
            seq_len = len(generated)
            pos_ids = torch.arange(0, seq_len, dtype=torch.long, device='cuda:0').unsqueeze(0)
            inp = torch.tensor([generated], device='cuda:0')
        else:
            cache_len = past_kv[0][0].shape[2]
            pos_ids = torch.tensor([[cache_len]], dtype=torch.long, device='cuda:0')
            inp = torch.tensor([[generated[-1]]], device='cuda:0')
        
        out = model(inp, position_ids=pos_ids, past_key_values=past_kv, use_cache=True, return_dict=True)
        
        # 🔍 详细检查 logits
        logits = out['logits'][0, -1]
        probs = logits.softmax(dim=-1)
        
        print(f"\nStep {i}:")
        print(f"  Logits stats: mean={logits.mean():.4f}, std={logits.std():.4f}, max={logits.max():.4f}, min={logits.min():.4f}")
        print(f"  Probs stats: max={probs.max():.4f}, entropy={-(probs * probs.log()).sum():.4f}")
        
        top10_probs, top10_indices = probs.topk(10)
        print(f"  Top 10 predictions:")
        for idx, (prob, token_id) in enumerate(zip(top10_probs, top10_indices)):
            token_str = tokenizer.decode([token_id.item()])
            print(f"    {idx+1}. [{token_id.item():6d}] '{token_str:20s}' (prob: {prob.item():.6f})")
        
        next_tok = logits.argmax().item()
        generated.append(next_tok)
        past_kv = out['past_key_values']

print(f"\nGenerated: {tokenizer.decode(generated)}")