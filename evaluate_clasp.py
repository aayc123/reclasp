import torch
from modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer
from decoding import infer

# 1. 加载模型
print("Loading model...")
model = LlamaForCausalLM.from_pretrained('/data/zn/model/models/Meta-Llama-3-8B', torch_dtype=torch.bfloat16)
model = model.to('cuda:0').eval()
tokenizer = AutoTokenizer.from_pretrained('/data/zn/model/models/Meta-Llama-3-8B')

# 2. 准备测试样本（三种方法）

# 方法 A: 使用现成的示例
xsum_example_1 = '''The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed. Repair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water.
Summary:'''

# 方法 B: 从数据集加载
# from datasets import load_dataset
# xsum_data = load_dataset('xsum', split='test')
# sample = xsum_data[0]
# xsum_example_2 = f"Article: {sample['document']}\nSummary:"

# 方法 C: 自定义文本
xsum_example_3 = '''Article: Apple announced its latest iPhone today at a special event in Cupertino. The new iPhone 15 features improved cameras, longer battery life, and a faster processor. Pre-orders begin next week with shipping starting in October.
Summary:'''

# 3. 选择一个示例运行
xsum_example = xsum_example_1  # 或 xsum_example_2, xsum_example_3

# 4. 运行 CLaSp 推理
print("Running CLaSp inference...")
result = infer(
    model, 
    tokenizer, 
    xsum_example, 
    generate_fn='clasp',  # 使用 CLaSp
    max_new_tokens=512,
    max_step_draft=12,
    num_skip_layers=24,
    update_interval=4,
    do_sample=False
)

# 5. 打印结果
print(f"\n{'='*60}")
print(f"Completion: {result['completion']}")
print(f"{'='*60}")
print(f"Time: {result['time']:.2f}s")
print(f"Matchness: {result['matchness']:.3f}")
print(f"Tokens generated: {result['generate_ids'].shape[1]}")