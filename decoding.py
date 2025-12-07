import torch
import torch.nn as nn
import time
#from transformers import top_k_top_p_filtering
from .dynamic_programming import DynamicLayerOptimizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
import torch
import torch.nn.functional as F

def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k: keep only top k tokens with highest probability (top-k filtering).
        top_p: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
        filter_value: value to replace filtered logits with
        min_tokens_to_keep: minimum number of tokens to keep
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    
    return logits

def clasp_generate(model, tokenizer, input_ids, max_new_tokens=10, early_stop=False,
                   max_step_draft=8, num_skip_layers=20, update_interval=4,
                   do_sample=False, top_k=0, top_p=0.85, temperature=0.2):
    """
    CLaSp: 动态层跳过的自推测解码
    """    
    step = 0
    update_counter = 0
    
    current_input_ids = input_ids
    generate_ids = torch.empty([input_ids.size(0), max_new_tokens + max_step_draft], 
                                dtype=torch.long, device=model.device)
    past_key_values = None
    
    # 初始化动态层优化器
    layer_optimizer = DynamicLayerOptimizer(model, num_skip_layers)
    
    # 初始跳过层（可以随机初始化或使用固定配置）
    # current_skip_layers = list(range(10, 30))  # 初始配置
    # 根据 num_skip_layers 动态计算
    num_layers = model.config.num_hidden_layers
    skip_ratio = num_skip_layers / num_layers
    # 均匀分布跳过层
    import numpy as np
    current_skip_layers = list(np.linspace(
        num_layers // 4, 
        num_layers * 3 // 4, 
        num_skip_layers, 
        dtype=int
    ))
    model.set_skip_layers(attn_skip_layer_id_set=current_skip_layers, mlp_skip_layer_id_set=[])
    
    n_matched = 0
    n_drafted = 0
    last_hidden_states = None  # 存储上次验证的隐藏状态
    step_accept_counts=[]
    with torch.no_grad():
        while True:
            if step >= max_new_tokens:
                break
            
            # ========== 阶段 1: 草稿生成 ==========
            if step == 0:
                # 第一个 token 使用完整模型
                seq_len = current_input_ids.shape[-1]
                position_ids = torch.arange(
                    0, seq_len, dtype=torch.long, device=model.device
                ).unsqueeze(0)
                output = model(input_ids=current_input_ids,
                               position_ids=position_ids,
                              past_key_values=past_key_values,
                              return_dict=True,
                              use_cache=True,
                              output_hidden_states=True)  # 需要隐藏状态
                logits = output['logits'][:, -1:]
                output_ids = sample(logits, do_sample=do_sample, top_k=top_k, 
                                   top_p=top_p, temperature=temperature)
                generate_ids[:, step] = output_ids
                current_input_ids = output_ids
                past_key_values = output['past_key_values']
                last_hidden_states = output['hidden_states']  # 保存隐藏状态
                step += 1
                
            else:
                # 使用当前跳过层配置生成草稿
                draft_current_input_ids = current_input_ids
                draft_past_key_values = past_key_values
                draft_tokens = [current_input_ids]
                
                for draft_step in range(max_step_draft):
                    # 1. 计算缓存长度 (cache_len) 作为当前 token 的起始位置
                    # 确保 draft_past_key_values 不为 None
                    if draft_past_key_values is not None and draft_past_key_values[0] is not None:
                        # 进一步检查 KV 缓存张量是否为 None
                        k_cache = draft_past_key_values[0][0] # k_cache 是第一个 Transformer 层的 Key cache 张量
                        cache_len = k_cache.shape[2] if k_cache is not None else 0
                    else:
                        cache_len = 0
                    
                    # 2. 计算 position_ids：对于单个 token，其位置是 cache_len
                    # 确保 position_ids 是一个张量 (1, 1)
                    draft_position_ids = torch.arange(
                        cache_len, cache_len + 1, dtype=torch.long, device=model.device
                    ).unsqueeze(0) # (1, 1) 张量
                    
                    with model.self_draft():
                        draft_output = model(input_ids=draft_current_input_ids,
                                            position_ids=draft_position_ids, 
                                            past_key_values=draft_past_key_values,
                                            return_dict=True,
                                            use_cache=True)
                    
                    draft_output_ids = sample(draft_output['logits'], 
                                             do_sample=do_sample, top_k=top_k, 
                                             top_p=top_p, temperature=temperature)
                    draft_tokens.append(draft_output_ids)
                    draft_current_input_ids = draft_output_ids
                    draft_past_key_values = draft_output['past_key_values']
                    
                    if step + draft_step + 2 >= max_new_tokens:
                        break
                
                drafted_n_tokens = len(draft_tokens) - 1
                drafted_input_ids = torch.cat(draft_tokens, dim=1)
                
                # ========== 阶段 2: 验证 ==========
                # 假设 input_ids 是 (batch_size, seq_len)
                # past_key_values 存储了 prompt 和之前生成的 token 的 KV 缓存
                # 我们可以根据 past_key_values 的长度来确定起始位置
                
                # 获取 KV 缓存的长度 (即已处理的总长度)
                cache_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
                # drafted_input_ids 的长度
                seq_len = drafted_input_ids.shape[-1]
                
                # 计算 position_ids，从 cache_len 开始
                position_ids = torch.arange(
                    cache_len, cache_len + seq_len, dtype=torch.long, device=model.device
                ).unsqueeze(0) # (1, seq_len)
                output = model(input_ids=drafted_input_ids,
                              position_ids=position_ids, # ⬅️ 添加这行
                              past_key_values=past_key_values,
                              return_dict=True,
                              use_cache=True,
                              output_hidden_states=True)
                
                logits = output['logits']
                output_ids = sample(logits, do_sample=do_sample, top_k=top_k, 
                                   top_p=top_p, temperature=temperature)
                
                # 检查匹配
                max_matched = ((output_ids[:, :-1] != drafted_input_ids[:, 1:]).cumsum(-1) == 0).sum(-1).item() + 1
                if step > 0: 
                    step_accept_counts.append(max_matched)
                max_of_max_matched = output_ids.size(1)
                
                if max_of_max_matched != max_matched:
                    output_ids = output_ids[:, :max_matched]
                    past_key_values = [
                        (k[:, :, :-(max_of_max_matched - max_matched)], 
                         v[:, :, :-(max_of_max_matched - max_matched)]) 
                        for k, v in output['past_key_values']
                    ]
                else:
                    past_key_values = output['past_key_values']
                
                generate_ids[:, step:step + output_ids.size(1)] = output_ids
                current_input_ids = output_ids[:, -1:]
                step += output_ids.size(1)
                
                n_matched += max_matched - 1
                n_drafted += drafted_n_tokens
                
                # ========== 阶段 3: 层优化 (CLaSp 特有!) ==========
                last_hidden_states = output['hidden_states']  # 更新隐藏状态
                update_counter += 1
                
                # 根据 update_interval 决定是否更新跳过层
                if update_counter % update_interval == 0:
                    # 提取最后接受 token 的所有层隐藏状态
                    last_accepted_hidden = [h[0, max_matched - 1, :] 
                                           for h in last_hidden_states]
                    last_accepted_hidden = torch.stack(last_accepted_hidden, dim=0)
                    
                    # 动态规划优化跳过层
                    new_skip_layers = layer_optimizer.optimize_skip_layers(
                        last_accepted_hidden
                    )
                    
                    # 更新模型的跳过层配置
                    model.set_skip_layers(attn_skip_layer_id_set=new_skip_layers, 
                                         mlp_skip_layer_id_set=[])
                    current_skip_layers = new_skip_layers
                    
                    # print(f"Updated skip layers: {len(new_skip_layers)} layers")
            
            if early_stop and tokenizer.eos_token_id in output_ids[0].tolist():
                break
    
    step = min(step, max_new_tokens)
    generate_ids = generate_ids[:, :step]
    full_sequence = torch.cat([input_ids, generate_ids], dim=1)
    return {
        'generate_ids': full_sequence,
        'matchness': n_matched / n_drafted if n_drafted > 0 else 0,
        'num_drafted_tokens': n_drafted,
        'accept_counts': step_accept_counts
    }

# 添加到映射
#generate_fn_mapping['clasp'] = clasp_generate

def sample(logits, return_probs: bool=False, do_sample: bool=False, top_k: int=50, top_p: float=0.7, temperature: float=0.7):

    if return_probs:

        all_probs = logits.softmax(-1)
        if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
            _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
            output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
            probs = torch.gather(all_probs, -1, output_ids.unsqueeze(-1)).squeeze(-1)
        else:
            probs, output_ids = torch.max(all_probs, dim=-1)
            
        return output_ids, probs

    else:

        if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
            _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
            output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
        else:
            output_ids = torch.argmax(logits, dim=-1)
            
        return output_ids

def base_generate(model, tokenizer, input_ids, max_new_tokens=10, 
                  do_sample=False, top_k=0, top_p=0.85, temperature=0.2,
                  early_stop=False):

    current_input_ids = input_ids
    generate_ids = torch.empty([input_ids.size(0), max_new_tokens], dtype=torch.long, device=model.device)
    past_key_values = None
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            if past_key_values is None:
                # 第一次调用：处理整个 prompt
                seq_len = current_input_ids.shape[-1]
                position_ids = torch.arange(0, seq_len, dtype=torch.long, device=model.device).unsqueeze(0)
            else:
                # 后续调用：只处理单个 token
                cache_len = past_key_values[0][0].shape[2]
                position_ids = torch.tensor([[cache_len]], dtype=torch.long, device=model.device)
            
            output = model(input_ids=current_input_ids,
                           position_ids=position_ids,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True)
            logits = output['logits'][:,-1:]
            output_ids = sample(logits, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
            generate_ids[:, step] = output_ids
            current_input_ids = output_ids
            past_key_values = output['past_key_values']

            if early_stop and current_input_ids.item() == tokenizer.eos_token_id:
                break

    step = min(step+1, max_new_tokens)
    generate_ids = generate_ids[:, :step]
                
    return {
        'generate_ids': generate_ids,
    }

def exact_self_speculative_generate(model, tokenizer, input_ids, max_new_tokens=10, early_stop=False,
                 max_step_draft=8, th_stop_draft=0.8, auto_th_stop_draft=True, auto_parameters=[1,0.5,0.9,1e-2,0.9],
                 do_sample=False, do_sample_draft=False, top_k=0, top_p=0.85, temperature=0.2):
    
    step = 0
    step_draft = 0
    step_verify = 0
    
    current_input_ids = input_ids
    generate_ids = torch.empty([input_ids.size(0), max_new_tokens+max_step_draft], dtype=torch.long, device=model.device)
    draft_generate_ids = torch.empty([input_ids.size(0), max_step_draft+1], dtype=torch.long, device=model.device)
    past_key_values = None

    n_matched = 0
    n_drafted = 0
    tmp_n_matched = 0
    tmp_n_drafted = 0
    tmp_matchness = 0
    with torch.no_grad():

        while True:

            if step >= max_new_tokens:
                break

            if step == 0:
                # first token use full model
                output = model(input_ids=current_input_ids,
                        past_key_values=past_key_values,
                        return_dict=True,
                        use_cache=True)
                logits = output['logits']
                logits = logits[:,-1:]
                output_ids = sample(logits, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
                generate_ids[:, step] = output_ids
                current_input_ids = output_ids
                past_key_values = output['past_key_values']
                last_hidden_states = output['hidden_states']
                step += 1

            else:
                
                draft_current_input_ids = current_input_ids
                draft_past_key_values = past_key_values
                draft_generate_ids[:, 0] = current_input_ids
                for step_draft in range(max_step_draft):
                    with model.self_draft():
                        draft_output = model(input_ids=draft_current_input_ids,
                            past_key_values=draft_past_key_values,
                            return_dict=True,
                            use_cache=True)
                    draft_probs = draft_output['logits'].softmax(-1)
                    draft_output_ids, draft_output_probs = sample(
                        draft_output['logits'], return_probs=True, do_sample=do_sample_draft, top_k=top_k, top_p=top_p, temperature=temperature)
                    draft_generate_ids[:, step_draft+1] = draft_output_ids
                    draft_current_input_ids = draft_output_ids
                    draft_past_key_values = draft_output['past_key_values']

                    if draft_output_probs.item() < th_stop_draft or step + step_draft + 2 >= max_new_tokens:
                        break
                
                drafted_n_tokens = step_draft + 1
                drafted_input_ids = draft_generate_ids[:, :drafted_n_tokens+1] # raft input + raft completion

                output = model(input_ids=drafted_input_ids,
                        past_key_values=past_key_values,
                        return_dict=True,
                        use_cache=True)
                logits = output['logits']
                # output_ids = torch.argmax(logits, dim=-1)
                output_ids = sample(logits, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)

                past_key_values = output['past_key_values']

                # including the one generated by the base model
                max_matched = ((output_ids[:, :-1] != drafted_input_ids[:, 1:]).cumsum(-1) == 0).sum(-1).item() + 1
                max_of_max_matched = output_ids.size(1)

                if max_of_max_matched != max_matched:
                    output_ids = output_ids[:, :max_matched]
                    
                    past_key_values = [
                        (k[:, :, :-(max_of_max_matched - max_matched)], v[:, :, :-(max_of_max_matched - max_matched)]) for k, v in past_key_values
                    ]

                generate_ids[:, step:step+output_ids.size(1)] = output_ids
                current_input_ids = output_ids[:, -1:]

                step += output_ids.size(1)

                # remove one generated by the base model
                n_matched += max_matched - 1
                n_drafted += drafted_n_tokens
                tmp_n_matched += max_matched - 1
                tmp_n_drafted += drafted_n_tokens
                step_verify += 1

                if auto_th_stop_draft and step_verify % auto_parameters[0] == 0:
                    tmp_matchness = auto_parameters[1]*(tmp_matchness) + (1-auto_parameters[1])*((max_matched - 1)/drafted_n_tokens)
                    if tmp_matchness<auto_parameters[2]:
                        new_th_stop_draft = th_stop_draft+auto_parameters[3]
                    else:
                        if drafted_n_tokens==max_step_draft:
                            new_th_stop_draft = th_stop_draft
                        else:
                            new_th_stop_draft = th_stop_draft-auto_parameters[3]
                    th_stop_draft = auto_parameters[4] * th_stop_draft + (1-auto_parameters[4]) * new_th_stop_draft
                    # print('draft_output_probs: {:.4f}, th_stop_draft: {:.4f}, tmp_matchness: {:.2f}, drafted_n_tokens: {:d}'.format(
                    #     draft_output_probs.item(), th_stop_draft, tmp_matchness, drafted_n_tokens))

            if early_stop and tokenizer.eos_token_id in output_ids[0].tolist():
                break

    step = min(step, max_new_tokens)
    generate_ids = generate_ids[:, :step]
            
    return {
        'generate_ids': generate_ids,
        'matchness': n_matched/n_drafted,
        'num_drafted_tokens': n_drafted,
        'th_stop_draft': th_stop_draft,
    }


def max_fn(x, eps=1e-6):
    x_max = torch.where(x > 0, x, 0)
    return x_max / (torch.sum(x_max) + eps)

def self_speculative_sample(model, tokenizer, input_ids, max_new_tokens=10, early_stop=False,
                 max_step_draft=8, th_stop_draft=0.5, th_random_draft=1.0, auto_th_stop_draft=True, 
                 auto_parameters=[1,0.5,0.9,1e-2,0.9],
                 do_sample=False, do_sample_draft=False, 
                 top_k=0, top_p=0.85, temperature=0.2):
    
    step = 0
    step_draft = 0
    step_verify = 0
    
    current_input_ids = input_ids
    generate_ids = torch.empty([input_ids.size(0), max_new_tokens+max_step_draft], 
                                dtype=torch.long, device=model.device)
    draft_generate_ids = torch.empty([input_ids.size(0), max_step_draft+2], 
                                      dtype=torch.long, device=model.device)
    draft_generate_probs = torch.empty([input_ids.size(0), max_step_draft, model.config.vocab_size], 
                                        dtype=torch.float, device=model.device)
    past_key_values = None

    n_matched = 0
    n_drafted = 0
    tmp_matchness = 0
    
    # 🔍 添加调试计数器
    debug_round = 0
    
    with torch.no_grad():
        while True:
            if step >= max_new_tokens:
                break

            if step == 0:
                # 第一个 token
                seq_len = current_input_ids.shape[-1]
                position_ids = torch.arange(
                    0, seq_len, dtype=torch.long, device=model.device
                ).unsqueeze(0)
                
                output = model(
                    input_ids=current_input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True
                )
                
                logits = output['logits'][:, -1:]
                output_ids = sample(logits, do_sample=do_sample, top_k=top_k, 
                                   top_p=top_p, temperature=temperature)
                
                # 🔍 打印第一个 token
                print(f"\n{'='*60}")
                print(f"🎯 First token generated: {tokenizer.decode(output_ids[0])}")
                print(f"{'='*60}\n")
                
                generate_ids[:, step] = output_ids
                current_input_ids = output_ids
                past_key_values = output['past_key_values']
                step += 1

            else:
                debug_round += 1
                print(f"\n{'─'*60}")
                print(f"🔄 Round {debug_round}: Drafting + Verification")
                print(f"{'─'*60}")
                
                # ========== DRAFT 阶段 ==========
                draft_current_input_ids = current_input_ids
                draft_past_key_values = past_key_values
                draft_generate_ids[:, 0] = current_input_ids
                random_list = torch.rand(max_step_draft)
                
                draft_tokens_decoded = []
                
                for step_draft in range(max_step_draft):
                    cache_len = draft_past_key_values[0][0].shape[2] if draft_past_key_values is not None else 0
                    draft_position_ids = torch.arange(
                        cache_len, cache_len + 1, dtype=torch.long, device=model.device
                    ).unsqueeze(0)
                    
                    with model.self_draft():
                        draft_output = model(
                            input_ids=draft_current_input_ids,
                            position_ids=draft_position_ids,
                            past_key_values=draft_past_key_values,
                            return_dict=True,
                            use_cache=True
                        )
                
                    if do_sample_draft and top_k != 1 and top_p != 0.0 and temperature != 0.0:
                        logits = draft_output['logits']
                        _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, 
                                                        top_k=top_k, top_p=top_p)
                        draft_probs = _logits.unsqueeze(1).softmax(-1)
                        draft_output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
                    else:
                        draft_probs = draft_output['logits'].softmax(-1)
                        draft_output_ids, _ = sample(draft_output['logits'], return_probs=True, 
                                                     do_sample=do_sample_draft)
                    
                    draft_generate_ids[:, step_draft+1] = draft_output_ids
                    draft_generate_probs[:, step_draft] = draft_probs
                    
                    # 🔍 记录 draft token
                    draft_token_text = tokenizer.decode(draft_output_ids[0])
                    draft_tokens_decoded.append(draft_token_text)
                    
                    draft_current_input_ids = draft_output_ids
                    draft_past_key_values = draft_output['past_key_values']
                    
                    origin_output_probs = torch.gather(draft_output['logits'].softmax(-1), -1, 
                                                       draft_output_ids.unsqueeze(-1)).squeeze(-1)
                    
                    # 🔍 打印每个 draft token 的信息
                    if step_draft < 3:  # 只打印前3个避免刷屏
                        print(f"  📝 Draft {step_draft+1}: '{draft_token_text}' (prob: {origin_output_probs.item():.4f})")
                    
                    if (origin_output_probs.item() < th_stop_draft and 
                        (1-random_list[step_draft]) <= th_random_draft) or \
                       step + step_draft + 2 >= max_new_tokens:
                        break
                
                drafted_n_tokens = step_draft + 1
                drafted_input_ids = draft_generate_ids[:, :drafted_n_tokens+1]
                
                # 🔍 打印 draft 总结
                print(f"  ✅ Drafted {drafted_n_tokens} tokens: {''.join(draft_tokens_decoded[:5])}...")
                
                # ========== VERIFY 阶段 ==========
                cache_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
                seq_len = drafted_input_ids.shape[-1]
                position_ids = torch.arange(
                    cache_len, cache_len + seq_len, dtype=torch.long, device=model.device
                ).unsqueeze(0)
                
                output = model(
                    input_ids=drafted_input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True
                )
                
                if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
                    logits = output['logits']
                    _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, 
                                                    top_k=top_k, top_p=top_p)
                    probs = _logits.unsqueeze(0).softmax(-1)
                else:
                    probs = output['logits'].softmax(-1)

                output_ids = draft_generate_ids[:, 1:drafted_n_tokens+2]
                observed_r_list = (probs[0, :drafted_n_tokens] / draft_generate_probs[0, :drafted_n_tokens]).cpu()
                
                # 🔍 验证过程
                accepted_tokens = []
                for i in range(drafted_n_tokens):
                    j = output_ids[0, i]
                    r = observed_r_list[i, j]
                    if random_list[i] < min(1, r):
                        accepted_tokens.append(tokenizer.decode([j.item()]))
                    else:
                        output_ids[0, i] = torch.multinomial(
                            max_fn((probs[0, i] - draft_generate_probs[0, i])), num_samples=1
                        )
                        accepted_tokens.append(f"[REJECT→{tokenizer.decode([output_ids[0, i].item()])}]")
                        break
                else:
                    i += 1
                    output_ids[0, i] = sample(output['logits'][0, i], do_sample=do_sample, 
                                             top_k=top_k, top_p=top_p, temperature=temperature)
                    accepted_tokens.append(f"[NEW:{tokenizer.decode([output_ids[0, i].item()])}]")

                max_matched = i + 1
                max_of_max_matched = drafted_input_ids.size(1)
                
                # 🔍 打印验证结果
                print(f"  🎯 Accepted: {max_matched-1}/{drafted_n_tokens} tokens")
                print(f"  📊 Tokens: {''.join(accepted_tokens[:5])}...")
                print(f"  ⏱️  Cumulative: {step} → {step + max_matched} tokens")

                if max_of_max_matched != max_matched:
                    output_ids = output_ids[:, :max_matched]
                    past_key_values = [
                        (k[:, :, :-(max_of_max_matched - max_matched)], 
                         v[:, :, :-(max_of_max_matched - max_matched)]) 
                        for k, v in past_key_values
                    ]
                else:
                    past_key_values = output['past_key_values']

                generate_ids[:, step:step+output_ids.size(1)] = output_ids
                current_input_ids = output_ids[:, -1:]
                step += output_ids.size(1)

                n_matched += max_matched - 1
                n_drafted += drafted_n_tokens
                step_verify += 1
                
                if auto_th_stop_draft and step_verify % auto_parameters[0] == 0:
                    tmp_matchness = auto_parameters[1]*(tmp_matchness) + \
                                   (1-auto_parameters[1])*((max_matched - 1)/drafted_n_tokens)
                    if tmp_matchness < auto_parameters[2]:
                        new_th_stop_draft = th_stop_draft + auto_parameters[3]
                    else:
                        if drafted_n_tokens == max_step_draft:
                            new_th_stop_draft = th_stop_draft
                        else:
                            new_th_stop_draft = th_stop_draft - auto_parameters[3]
                    th_stop_draft = auto_parameters[4] * th_stop_draft + \
                                   (1-auto_parameters[4]) * new_th_stop_draft

            if early_stop and tokenizer.eos_token_id in output_ids[0].tolist():
                print(f"\n🛑 Early stop triggered (EOS token found)\n")
                break

    step = min(step, max_new_tokens)
    generate_ids = generate_ids[:, :step]
    
    # 🔍 最终统计
    final_matchness = n_matched/n_drafted if n_drafted > 0 else 0
    print(f"\n{'='*60}")
    print(f"📊 Final Statistics:")
    print(f"   Total tokens generated: {step}")
    print(f"   Total drafted: {n_drafted}")
    print(f"   Total matched: {n_matched}")
    print(f"   Matchness: {final_matchness:.3f}")
    print(f"{'='*60}\n")
            
    return {
        'generate_ids': generate_ids,
        'matchness': final_matchness,
        'num_drafted_tokens': n_drafted,
        'th_stop_draft': th_stop_draft,
    }



generate_fn_mapping = {
    'base': base_generate,
    'exact_self_speculative_generate': exact_self_speculative_generate,
    'essg': exact_self_speculative_generate,
    'self_speculative_sample': self_speculative_sample,
    'sss': self_speculative_sample,
    'clasp': clasp_generate,
}

def infer(model, tokenizer, prompt, generate_fn='base', 
          decode_timing=True, seed=42, *args, **kargs):

    if isinstance(generate_fn, str):
        generate_fn = generate_fn_mapping[generate_fn]

    if seed is not None:
        torch.manual_seed(seed)
              
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    if decode_timing:
        tic = time.time()
    generate_dict = generate_fn(model, tokenizer, input_ids, *args, **kargs)
    generate_ids = generate_dict['generate_ids']
    if decode_timing:
        toc = time.time()
        decode_time = toc - tic
    else:
        decode_time = None
    completion = tokenizer.decode(generate_ids[0])
    generate_dict['completion'] = completion
    generate_dict['time'] = decode_time
    return generate_dict

def clip_input(tokenizer, prompt, task_name, max_new_tokens=512, prompt_shots=''):
    if task_name == 'xsum':
        input_ids = tokenizer(
            prompt_shots +'Article: ' + prompt['document'] + '\nSummary:',
            return_tensors='pt').input_ids
    elif task_name == 'cnndm':
        input_ids = tokenizer(
            prompt_shots +'Article: ' + prompt['article'] + '\nSummary:',
            return_tensors='pt').input_ids
    elif task_name == 'humaneval':
        format_tabs=True
        if format_tabs:
            prompt = prompt['prompt'].replace("    ", "\t")
        else:
            prompt = prompt['prompt']
        input_ids = tokenizer(prompt,return_tensors='pt').input_ids
    if len(input_ids[0])+max_new_tokens>=4096:
        print('(input ids+max token)>4096')
        sample_num = (len(input_ids[0])+max_new_tokens-4096) 
        input_ids = torch.cat((input_ids[0][:2],input_ids[0][2:-3][:-sample_num],input_ids[0][-3:]),dim=0).unsqueeze(0)
    return  input_ids
    
def infer_input_ids(model, tokenizer, input_ids, generate_fn='base', 
          decode_timing=True, seed=42, *args, **kargs):

    if isinstance(generate_fn, str):
        generate_fn = generate_fn_mapping[generate_fn]

    if seed is not None:
        torch.manual_seed(seed)
        
    input_ids = input_ids.to(model.device)
    if decode_timing:
        tic = time.time()
    generate_dict = generate_fn(model, tokenizer, input_ids, *args, **kargs)
    generate_ids = generate_dict['generate_ids']
    if decode_timing:
        toc = time.time()
        decode_time = toc - tic
    else:
        decode_time = None
    completion = tokenizer.decode(generate_ids[0, input_ids.size(0):],skip_special_tokens=True)
    generate_dict['completion'] = completion
    generate_dict['time'] = decode_time
    return generate_dict
