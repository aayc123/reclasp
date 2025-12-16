import torch
import torch.nn as nn
import time
#from transformers import top_k_top_p_filtering
from .dynamic_programming import DynamicLayerOptimizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
import torch
import torch.nn.functional as F
import numpy as np
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


import time
import torch
import torch.nn as nn
import time
from datetime import datetime
import json
import csv

def clasp_generate(model, tokenizer, input_ids, max_new_tokens=128, early_stop=False,
                   max_step_draft=8, num_skip_layers=15, update_interval=8,
                   do_sample=False, top_k=0, top_p=0.85, temperature=0.0,th_stop_draft=0.7,
                   log_file="clasp_timing_log.json"):
    """
    CLaSp: 动态层跳过的自推测解码
    添加详细的时间记录功能
    """
    # 初始化时间记录数据结构
    timing_records = {
        "metadata": {
            "model": model.__class__.__name__,
            "max_new_tokens": max_new_tokens,
            "max_step_draft": max_step_draft,
            "num_skip_layers": num_skip_layers,
            "update_interval": update_interval,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "rounds": []  # 记录每一轮的时间
    }
    
    # 创建CSV文件用于记录
    csv_filename = log_file.replace('.json', '.csv')
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['round', 'draft_time_ms', 'verify_time_ms', 'dp_time_ms', 
                         'drafted_tokens', 'accepted_tokens', 'skip_layers'])
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
    step = 0
    update_counter = 0
    n_matched = 0
    n_drafted = 0
    n_rounds = 0
    current_input_ids = input_ids
    generate_ids = torch.empty([input_ids.size(0), max_new_tokens + max_step_draft], 
                               dtype=torch.long, device=model.device)
    past_key_values = None
    
    fixed_skip_layers = [11, 14, 18, 22, 24, 25, 26]
    current_skip_layers= [10, 11, 24, 25, 27, 28]
    fixed_mlp_skip = [11, 14, 18, 22, 24, 25, 26]
    model.set_skip_layers(attn_skip_layer_id_set=fixed_skip_layers, 
                         mlp_skip_layer_id_set=fixed_mlp_skip)
    
    last_hidden_states = None
    step_accept_counts = []
    need_hidden_states = 0
    with torch.no_grad():
        while True:
            if step >= max_new_tokens:
                break
            
            # 初始化本轮时间记录
            round_timing = {
                "round": n_rounds + 1,
                "draft_time_ms": 0,
                "verify_time_ms": 0,
                "dp_time_ms": 0,
                "drafted_tokens": 0,
                "accepted_tokens": 0,
                "skip_layers": fixed_skip_layers if n_rounds == 0 else current_skip_layers
            }
            
            # ========== 阶段 1: 草稿生成 ==========
            if step == 0:
                # 第一个 token 使用完整模型
                round_start = time.perf_counter()
                
                seq_len = current_input_ids.shape[-1]
                position_ids = torch.arange(
                    0, seq_len, dtype=torch.long, device=model.device
                ).unsqueeze(0)
                output = model(input_ids=current_input_ids,
                               position_ids=position_ids,
                               past_key_values=past_key_values,
                               return_dict=True,
                               use_cache=True,
                               output_hidden_states=False)
                logits = output['logits'][:, -1:]
                output_ids = sample(logits, do_sample=do_sample, top_k=top_k, 
                                   top_p=top_p, temperature=temperature)
                
                draft_end = time.perf_counter()
                round_timing["draft_time_ms"] = (draft_end - round_start) * 1000
                round_timing["verify_time_ms"] = 0  # 第一轮没有验证
                round_timing["dp_time_ms"] = 0
                round_timing["drafted_tokens"] = 1
                round_timing["accepted_tokens"] = 1
                
                generate_ids[:, step] = output_ids
                current_input_ids = output_ids
                past_key_values = output['past_key_values']
                step += 1
                n_rounds += 1
                n_drafted += 1
                n_matched += 1
                
                # 记录本轮时间
                timing_records["rounds"].append(round_timing)
                csv_writer.writerow([
                    round_timing["round"],
                    round_timing["draft_time_ms"],
                    round_timing["verify_time_ms"],
                    round_timing["dp_time_ms"],
                    round_timing["drafted_tokens"],
                    round_timing["accepted_tokens"],
                    len(round_timing["skip_layers"])
                ])
                
            else:
                # ========== DRAFT 阶段计时 ==========
                draft_start = time.perf_counter()
                
                draft_current_input_ids = current_input_ids
                draft_kv_cache = past_key_values
                draft_tokens = [current_input_ids]
                draft_confidences = []
                early_stopped = False
                
                for draft_step in range(max_step_draft):
                    cache_len = past_key_values[0][0].shape[2]
                    pos_id = cache_len + draft_step
                    draft_position_ids = torch.tensor([[pos_id]], device=model.device)
                    
                    with model.self_draft():
                        draft_output = model(input_ids=draft_current_input_ids,
                                             position_ids=draft_position_ids,
                                             past_key_values=draft_kv_cache,
                                             return_dict=True,
                                             use_cache=True)
                    
                    # 使用增强的sample函数，同时获取token和置信度
                    draft_output_ids, confidence = sample_with_confidence(
                        draft_output['logits'][:, -1:],
                        return_probs=True,
                        do_sample=do_sample,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature
                    )
                    
                    # 记录置信度
                    conf_value = confidence.item() if confidence.numel() == 1 else confidence.mean().item()
                    draft_confidences.append(conf_value)
                    
                    draft_tokens.append(draft_output_ids)
                    draft_current_input_ids = draft_output_ids
                    draft_kv_cache = draft_output['past_key_values']
                    
                    # 检查是否应该提前停止
                    if conf_value < th_stop_draft:
                        early_stopped = True
                        # print(f"  ⚠️ Draft early stop at step {draft_step+1}: "
                        #       f"confidence {conf_value:.3f} < threshold {th_stop_draft}")
                        break
                    
                    if step + draft_step + 2 >= max_new_tokens:
                        break
                
                draft_end = time.perf_counter()
                draft_time = (draft_end - draft_start) * 1000
                round_timing["draft_time_ms"] = draft_time
                round_timing["early_stopped"] = early_stopped
                round_timing["min_confidence"] = min(draft_confidences) if draft_confidences else 1.0
                
                drafted_n_tokens = len(draft_tokens) - 1
                drafted_input_ids = torch.cat(draft_tokens, dim=1)
                round_timing["drafted_tokens"] = drafted_n_tokens
                
                # ========== VERIFY 阶段计时 ==========
                verify_start = time.perf_counter()
                verify_past_key_values = past_key_values
                cache_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
                seq_len = drafted_input_ids.shape[-1]
                position_ids = torch.arange(
                    cache_len, cache_len + seq_len, dtype=torch.long, device=model.device
                ).unsqueeze(0)
                flag=True if need_hidden_states>=1 else False
                output = model(input_ids=drafted_input_ids,
                               position_ids=position_ids,
                               past_key_values=past_key_values,
                               return_dict=True,
                               use_cache=True,
                               output_hidden_states=flag)
                
                logits = output['logits']
                output_ids = sample(logits, do_sample=do_sample, top_k=top_k, 
                                   top_p=top_p, temperature=temperature)
                
                # 检查匹配
                max_matched = ((output_ids[:, :-1] != drafted_input_ids[:, 1:]).cumsum(-1) == 0).sum(-1).item() + 1
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
                
                verify_end = time.perf_counter()
                verify_time = (verify_end - verify_start) * 1000  # 转换为毫秒
                round_timing["verify_time_ms"] = verify_time
                round_timing["accepted_tokens"] = max_matched - 1
                
                generate_ids[:, step:step + output_ids.size(1)] = output_ids
                current_input_ids = output_ids[:, -1:]
                step += output_ids.size(1)
                
                n_matched += max_matched - 1
                n_drafted += drafted_n_tokens
                n_rounds += 1
                step_accept_counts.append(max_matched)
                
                if max_matched - 1<=3:
                    need_hidden_states +=1
                else:
                    need_hidden_states=0
                # ========== DP 阶段计时 ==========
                dp_time = 0
                # if update_counter % update_interval == 0:
                if  need_hidden_states>=5 and update_counter<3:
                    update_counter += 1
                    # print("🔄 触发动态规划跳层优化")
                    # if(need_hidden_states==0):
                    #     continue
                    dp_start = time.perf_counter()
                    need_hidden_states=0
                    last_hidden_states = output['hidden_states']
                    last_accepted_hidden = [h[0, max_matched - 1, :] 
                                           for h in last_hidden_states]
                    last_accepted_hidden = torch.stack(last_accepted_hidden, dim=0)
                    
                    torch.cuda.synchronize()
                    layer_optimizer = DynamicLayerOptimizer(model, num_skip_layers)
                    new_skip_layers = layer_optimizer.optimize_skip_layers_v2(
                        last_accepted_hidden,
                        past_key_values
                    )
                    torch.cuda.synchronize()
                    
                    dp_end = time.perf_counter()
                    dp_time = (dp_end - dp_start) * 1000  # 转换为毫秒
                    
                    # 更新模型的跳过层配置
                    # if len(new_skip_layers) != num_skip_layers:
                    #     print(f"⚠️ DP返回层数 {len(new_skip_layers)} ≠ {num_skip_layers}，强制修正")
                    #     num_layers = model.config.num_hidden_layers
                    #     fixed_front = 10
                    #     fixed_back = 10
                    #     dynamic_range = list(range(fixed_front, num_layers - fixed_back))
                    #     step_size = len(dynamic_range) // num_skip_layers
                    #     new_skip_layers = dynamic_range[::step_size][:num_skip_layers]
                    
                    model.set_skip_layers(attn_skip_layer_id_set=new_skip_layers, 
                                         mlp_skip_layer_id_set=[])
                    current_skip_layers = new_skip_layers
                
                round_timing["dp_time_ms"] = dp_time
                round_timing["skip_layers"] = current_skip_layers if 'current_skip_layers' in locals() else fixed_skip_layers
                
                # 记录本轮时间
                timing_records["rounds"].append(round_timing)
                csv_writer.writerow([
                    round_timing["round"],
                    round_timing["draft_time_ms"],
                    round_timing["verify_time_ms"],
                    round_timing["dp_time_ms"],
                    round_timing["drafted_tokens"],
                    round_timing["accepted_tokens"],
                    len(round_timing["skip_layers"])
                ])
                
                # 打印当前轮次的时间统计
                # print(f"\n[Round {n_rounds}] Draft: {draft_time:.2f}ms | "
                #       f"Verify: {verify_time:.2f}ms | DP: {dp_time:.2f}ms | "
                #       f"Drafted: {drafted_n_tokens} | Accepted: {max_matched-1}")
            
            if early_stop and tokenizer.eos_token_id in output_ids[0].tolist():
                break
    
    # 记录总时间
    end_event.record()
    torch.cuda.synchronize()
    total_time = start_event.elapsed_time(end_event)  # 毫秒
    
    step = min(step, max_new_tokens)
    generate_ids = generate_ids[:, :step]
    full_sequence = torch.cat([input_ids, generate_ids], dim=1)
    
    # 计算统计指标
    tau = (n_matched + n_rounds) / n_rounds if n_rounds > 0 else 1.0
    matchness = n_matched / n_drafted if n_drafted > 0 else 0.0
    
    # 更新元数据
    timing_records["metadata"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timing_records["metadata"]["total_time_ms"] = total_time
    timing_records["metadata"]["total_tokens"] = step
    timing_records["metadata"]["n_rounds"] = n_rounds
    timing_records["metadata"]["n_drafted"] = n_drafted
    timing_records["metadata"]["n_matched"] = n_matched
    timing_records["metadata"]["tau"] = tau
    timing_records["metadata"]["matchness"] = matchness
    
    # 计算时间统计
    total_draft_time = sum(r["draft_time_ms"] for r in timing_records["rounds"])
    total_verify_time = sum(r["verify_time_ms"] for r in timing_records["rounds"])
    total_dp_time = sum(r["dp_time_ms"] for r in timing_records["rounds"])
    
    timing_records["statistics"] = {
        "total_draft_time_ms": total_draft_time,
        "total_verify_time_ms": total_verify_time,
        "total_dp_time_ms": total_dp_time,
        "avg_draft_time_ms": total_draft_time / len(timing_records["rounds"]) if timing_records["rounds"] else 0,
        "avg_verify_time_ms": total_verify_time / len(timing_records["rounds"]) if timing_records["rounds"] else 0,
        "avg_dp_time_ms": total_dp_time / sum(1 for r in timing_records["rounds"] if r["dp_time_ms"] > 0) if any(r["dp_time_ms"] > 0 for r in timing_records["rounds"]) else 0
    }
    
    # 保存JSON文件
    with open(log_file, 'w') as f:
        json.dump(timing_records, f, indent=2, ensure_ascii=False)
    
    csv_file.close()
    
    # 打印总结
    # print(f"\n{'='*60}")
    # print(f"CLaSp生成完成:")
    # print(f"  总tokens: {step}")
    # print(f"  总时间: {total_time:.2f}ms")
    # print(f"  Draft轮数: {n_rounds}")
    # print(f"  总draft tokens: {n_drafted}")
    # print(f"  总接受tokens: {n_matched}")
    # print(f"  τ (平均接受长度): {tau:.3f}")
    # print(f"  Matchness (接受率): {matchness:.3f}")
    # print(f"\n时间统计:")
    # print(f"  Draft总时间: {total_draft_time:.2f}ms")
    # print(f"  Verify总时间: {total_verify_time:.2f}ms")
    # print(f"  DP总时间: {total_dp_time:.2f}ms")
    # print(f"  JSON日志保存到: {log_file}")
    # print(f"  CSV日志保存到: {csv_filename}")
    # print(f"{'='*60}\n")
    
    return {
        'generate_ids': full_sequence,
        'matchness': matchness,
        'num_drafted_tokens': n_drafted,
        'accept_counts': step_accept_counts,
        'timing_records': timing_records,  # 返回时间记录
        'total_time_ms': total_time
    }

# 添加到映射
#generate_fn_mapping['clasp'] = clasp_generate

def sample_with_confidence(logits, return_probs=True, do_sample=False, top_k=50, top_p=0.7, temperature=0.7):
    # 获取原始概率分布（用于计算置信度）
    original_probs = F.softmax(logits, dim=-1)
    
    if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
        # 采样模式
        _logits = top_k_top_p_filtering(
            logits.view(-1, logits.size(-1)) / temperature, 
            top_k=top_k, 
            top_p=top_p
        )
        filtered_probs = F.softmax(_logits, dim=-1)
        output_ids = torch.multinomial(filtered_probs, num_samples=1).view(logits.shape[:-1])
    else:
        # 贪婪解码
        output_ids = torch.argmax(logits, dim=-1)
    
    # 获取选中token的原始概率作为置信度
    if output_ids.dim() == 1:
        output_ids_expanded = output_ids.unsqueeze(-1)
    else:
        output_ids_expanded = output_ids.unsqueeze(-1)
    
    confidence = torch.gather(original_probs, -1, output_ids_expanded).squeeze(-1)
    
    if return_probs:
        return output_ids, confidence
    else:
        return output_ids
    
    
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

def base_generate(model, tokenizer, input_ids, max_new_tokens=128, 
                  do_sample=False, top_k=0, top_p=0.85, temperature=0.0,
                  early_stop=False):

    current_input_ids = input_ids
    gen_list = []
    past_key_values = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if past_key_values is None:
                seq_len = current_input_ids.shape[-1]
                position_ids = torch.arange(0, seq_len, dtype=torch.long, device=model.device).unsqueeze(0)
            else:
                cache_len = past_key_values[0][0].shape[2]
                position_ids = torch.tensor([[cache_len]], dtype=torch.long, device=model.device)

            output = model(input_ids=current_input_ids,
                           position_ids=position_ids,
                           past_key_values=past_key_values,
                           return_dict=True,
                           use_cache=True)
            logits = output['logits'][:, -1:]
            output_ids = sample(logits, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)

            gen_list.append(output_ids)               # collect generated tokens
            current_input_ids = output_ids
            past_key_values = output['past_key_values']

            if early_stop and current_input_ids.item() == tokenizer.eos_token_id:
                break

    if len(gen_list) == 0:
        gen_ids = torch.empty((input_ids.size(0), 0), dtype=torch.long, device=model.device)
    else:
        gen_ids = torch.cat(gen_list, dim=1)  # (batch, n_generated)

    full_sequence = torch.cat([input_ids, gen_ids], dim=1)
    return {
        'generate_ids': full_sequence,       # 与 clasp_generate 保持一致
        'num_new_tokens': gen_ids.size(1),
    }


def exact_self_speculative_generate(model, tokenizer, input_ids, max_new_tokens=10, early_stop=False,
                 max_step_draft=8, th_stop_draft=0.8, auto_th_stop_draft=True, auto_parameters=[1,0.5,0.9,1e-2,0.9],
                 do_sample=False, do_sample_draft=False, top_k=0, top_p=0.85, temperature=0.0):
    
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

def self_speculative_sample(model, tokenizer, input_ids, max_new_tokens=128, early_stop=False,
                 max_step_draft=8, th_stop_draft=0.5, th_random_draft=1.0, auto_th_stop_draft=True, 
                 auto_parameters=[1,0.5,0.9,1e-2,0.9],
                 do_sample=False, do_sample_draft=False, 
                 top_k=0, top_p=0.85, temperature=0.0):
    
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
    n_verification_rounds=0
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
                n_verification_rounds+=1
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
                tau = (n_matched + n_verification_rounds) / n_verification_rounds
            if early_stop and tokenizer.eos_token_id in output_ids[0].tolist():
                print(f"\n🛑 Early stop triggered (EOS token found)\n")
                break
    final_tau = (n_matched + n_verification_rounds) / n_verification_rounds if n_verification_rounds > 0 else 0
    step = min(step, max_new_tokens)
    generate_ids = generate_ids[:, :step]
    full_sequence = torch.cat([input_ids, generate_ids], dim=1)
    # 🔍 最终统计
    final_matchness = n_matched/n_drafted if n_drafted > 0 else 0
    print(f"\n{'='*60}")
    print(f"📊 Final Statistics:")
    print(f"   Total tokens generated: {step}")
    print(f"   Total drafted: {n_drafted}")
    print(f"   Total matched: {n_matched}")
    print(f"   Matchness: {final_matchness:.3f}")
    print(f"   Matchness: {final_matchness:.3f}")
    print(f"{'='*60}\n")
            
    return {
        'generate_ids': full_sequence,
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
