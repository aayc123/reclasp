import torch
import torch.nn.functional as F

class DynamicLayerOptimizer:
    def __init__(self, model, skip_threshold=0.3, max_skip_layers=None):
        self.model = model
        self.skip_threshold = skip_threshold
        self.num_layers = model.config.num_hidden_layers
        self.hidden_size = model.config.hidden_size
        self.dtype = model.dtype
        self.device = model.device
        
        if self.dtype in [torch.float16, torch.bfloat16]:
            self.neg_inf = -65000.0
        else:
            self.neg_inf = -1e9
        
        self.max_skip_layers = max_skip_layers if max_skip_layers is not None else self.num_layers

    @torch.inference_mode()
    def optimize_skip_layers_v2(self, last_hidden_states, past_key_values):
        """
        第四轮优化（保守版）：在 20ms 基础上做微调
        目标：从 20ms 优化到 10-15ms
        """
        L = self.num_layers
        
        # ====== 优化1：候选层筛选（保持原逻辑但微调）======
        fixed_front = 10
        fixed_back = 10
        candidate_range = list(range(fixed_front, L - fixed_back))
        
        # 候选层数量：在准确性和速度之间平衡
        # 10 层是个甜蜜点
        max_candidates = 10
        step = max(1, len(candidate_range) // max_candidates)
        candidate_layers = candidate_range[::step][:max_candidates]
        
        # ====== 步骤1：批量前向传播（保持原有逻辑）======
        executed_states = self._batch_forward_selective_layers_optimized(
            last_hidden_states[0], 
            past_key_values,
            candidate_layers
        )
        
        num_candidates = len(candidate_layers)
        
        # ====== 步骤2：优化的相似度计算（一次性完成）======
        # 预先 normalize 所有向量
        targets_norm = F.normalize(last_hidden_states[1:L+1], p=2, dim=-1)
        executed_norm = F.normalize(executed_states, p=2, dim=-1)
        
        # 计算 execute 相似度
        sim_execute = torch.zeros(L, device=self.device, dtype=torch.float32)
        for idx, layer_id in enumerate(candidate_layers):
            sim_execute[layer_id] = (executed_norm[idx] * targets_norm[layer_id]).sum(dim=-1)
        
        # 构建状态矩阵
        all_states = torch.zeros((num_candidates+1, self.hidden_size), device=self.device, dtype=self.dtype)
        all_states[0] = last_hidden_states[0]
        all_states[1:] = executed_states
        all_states_norm = F.normalize(all_states, p=2, dim=-1)
        
        # 预计算 skip 相似度矩阵（一次矩阵乘法）
        candidate_targets_norm = targets_norm[candidate_layers]
        sim_skip = torch.mm(all_states_norm, candidate_targets_norm.t())  # [num_candidates+1, num_candidates]
        
        # ====== 步骤3：完全向量化的 DP（关键优化）======
        K = min(self.max_skip_layers + 1, num_candidates + 1)
        dp = torch.full((num_candidates+1, K), self.neg_inf, device=self.device, dtype=torch.float32)
        state_idx = torch.full((num_candidates+1, K), -1, dtype=torch.long, device=self.device)
        
        # 初始化
        dp[0, 0] = 0.0
        state_idx[0, 0] = 0
        
        # 向量化 DP 更新
        for i in range(num_candidates):
            layer_id = candidate_layers[i]
            
            # 找到有效状态（向量化）
            valid_mask = dp[i] > self.neg_inf + 1000
            if not valid_mask.any():
                continue
            
            valid_j = torch.where(valid_mask)[0]
            curr_state_idx = state_idx[i, valid_j]
            curr_dp = dp[i, valid_j]
            
            # === 执行层更新（完全向量化）===
            new_sim_exec = curr_dp + sim_execute[layer_id]
            exec_better = new_sim_exec > dp[i+1, valid_j]
            
            # 原地更新（避免创建新 tensor）
            dp[i+1, valid_j] = torch.where(exec_better, new_sim_exec, dp[i+1, valid_j])
            state_idx[i+1, valid_j] = torch.where(exec_better, 
                                                   torch.full_like(valid_j, i + 1), 
                                                   state_idx[i+1, valid_j])
            
            # === 跳过层更新（完全向量化）===
            skip_valid = valid_j[valid_j < K - 1]
            if len(skip_valid) > 0:
                # 批量查询相似度
                skip_state_idx = state_idx[i, skip_valid]
                skip_sims = sim_skip[skip_state_idx, i]
                
                new_sim_skip = curr_dp[:len(skip_valid)] + skip_sims + self.skip_threshold
                skip_better = new_sim_skip > dp[i+1, skip_valid + 1]
                
                dp[i+1, skip_valid + 1] = torch.where(skip_better, new_sim_skip, dp[i+1, skip_valid + 1])
                state_idx[i+1, skip_valid + 1] = torch.where(skip_better, 
                                                              skip_state_idx, 
                                                              state_idx[i+1, skip_valid + 1])
        
        # 找到最佳策略
        best_j = torch.argmax(dp[num_candidates])
        
        # 回溯决策
        skip_layers = []
        curr_i = num_candidates
        curr_j = best_j.item()
        
        while curr_i > 0:
            if curr_j > 0 and state_idx[curr_i, curr_j] == state_idx[curr_i-1, curr_j-1]:
                skip_layers.append(candidate_layers[curr_i - 1])
                curr_j -= 1
            curr_i -= 1
        
        skip_layers.reverse()
        return skip_layers, dp[num_candidates, best_j].item()
    
    def _batch_forward_selective_layers_optimized(self, initial_hidden, past_key_values, layer_indices):
        """
        优化版本：减少不必要的 clone 和内存分配
        """
        if len(layer_indices) == 0:
            return torch.empty((0, self.hidden_size), device=self.device, dtype=self.dtype)
        
        executed_states = []
        current_state = initial_hidden
        
        # 按层 ID 排序确保顺序前向
        prev_idx = 0
        for target_layer in layer_indices:
            # 执行到目标层
            for i in range(prev_idx, target_layer + 1):
                current_state = self._forward_layer_fast(i, current_state, past_key_values[i])
            
            # 保存输出（只在需要时 clone）
            executed_states.append(current_state)
            prev_idx = target_layer + 1
        
        # 一次性 stack（比逐个 append 快）
        return torch.stack(executed_states, dim=0)
    
    def _forward_layer_fast(self, layer_idx, hidden_state, past_key_value):
        """
        快速单层前向（内联优化）
        """
        layer = self.model.model.layers[layer_idx]
        
        # 最小化 tensor 创建
        h_seq = hidden_state.view(1, 1, -1)  # 使用 view 而非 unsqueeze
        
        past_len = past_key_value[0].shape[2]
        position_ids = torch.tensor([[past_len]], dtype=torch.long, device=self.device)
        
        # 简化 mask 创建
        total_len = past_len + 1
        mask = torch.zeros((1, 1, 1, total_len), device=self.device, dtype=self.dtype)
        
        output = layer(
            h_seq,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=False
        )
        
        return output[0].view(-1)  # 使用 view 而非 squeeze
    
    def _batch_forward_selective_layers(self, initial_hidden, past_key_values, layer_indices):
        """保留原始版本作为备份"""
        executed_states = []
        current_state = initial_hidden
        
        prev_idx = 0
        for layer_id in layer_indices:
            for i in range(prev_idx, layer_id + 1):
                layer = self.model.model.layers[i]
                h_seq = current_state.unsqueeze(0).unsqueeze(0)
                
                past_len = past_key_values[i][0].shape[2]
                position_ids = torch.tensor([[past_len]], dtype=torch.long, device=self.device)
                
                total_len = past_len + 1
                mask = torch.zeros((1, 1, 1, total_len), device=self.device, dtype=self.dtype)
                
                output = layer(
                    h_seq,
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values[i],
                    use_cache=False
                )
                
                current_state = output[0].squeeze(0).squeeze(0)
            
            executed_states.append(current_state.clone())
            prev_idx = layer_id + 1
        
        return torch.stack(executed_states, dim=0) if executed_states else torch.empty((0, self.hidden_size), device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def optimize_skip_layers(self, last_hidden_states, past_key_values):
        """保留原始版本"""
        L = self.num_layers
        decisions = []
        layer_similarities = []
        current_state = last_hidden_states[0].clone()
        
        for i in range(L):
            executed_state = self._forward_layer(
                layer_idx=i,
                hidden_state=current_state,
                past_key_value=past_key_values[i]
            )
            
            target = F.normalize(last_hidden_states[i+1], p=2, dim=-1)
            norm_executed = F.normalize(executed_state, p=2, dim=-1)
            norm_current = F.normalize(current_state, p=2, dim=-1)
            
            sim_execute = (norm_executed * target).sum(dim=-1)
            sim_skip = (norm_current * target).sum(dim=-1)
            
            remaining_skips = self.max_skip_layers - sum(decisions)
            if remaining_skips > 0:
                dynamic_threshold = self.skip_threshold * (1.0 + 0.1 * remaining_skips)
            else:
                dynamic_threshold = self.skip_threshold
                
            do_skip = (sim_skip + dynamic_threshold) > sim_execute
            
            if sum(decisions) >= self.max_skip_layers:
                do_skip = False
            
            if do_skip:
                decisions.append(True)
            else:
                decisions.append(False)
                current_state = executed_state
            
            layer_similarities.append({
                'execute': sim_execute.item(),
                'skip': sim_skip.item(),
                'decision': do_skip.item()
            })
        
        skip_layers = [i for i, skip in enumerate(decisions) if skip]
        return skip_layers, layer_similarities

    def _forward_layer(self, layer_idx, hidden_state, past_key_value):
        """单层前向传播（备用）"""
        layer = self.model.model.layers[layer_idx]
        h_seq = hidden_state.unsqueeze(0).unsqueeze(0)
        
        past_len = past_key_value[0].shape[2]
        position_ids = torch.tensor([[past_len]], dtype=torch.long, device=self.device)
        
        total_len = past_len + 1
        mask = torch.zeros((1, 1, 1, total_len), device=self.device, dtype=self.dtype)
        
        output = layer(
            h_seq,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=False
        )
        
        return output[0].squeeze(0).squeeze(0)
#     @torch.inference_mode()
#     def optimize_skip_layers(self, last_hidden_states, past_key_values):
#         """
#         Args:
#             last_hidden_states: [L+1, hidden_size] 目标隐藏状态 (Ground Truth)
#             past_key_values: 每一层的 KV Cache，必须传入！
#         """
#         L = self.num_layers
        
#         # 存储每个层的决策和相似度
#         decisions = []  # True表示跳过该层
#         layer_similarities = []
        
#         # 初始状态
#         current_state = last_hidden_states[0].clone()
        
#         for i in range(L):
#             # 1. 计算执行当前层后的状态
#             executed_state = self._forward_layer(
#                 layer_idx=i,
#                 hidden_state=current_state,
#                 past_key_value=past_key_values[i]
#             )
            
#             # 2. 计算相似度
#             target = F.normalize(last_hidden_states[i+1], p=2, dim=-1)
            
#             norm_executed = F.normalize(executed_state, p=2, dim=-1)
#             norm_current = F.normalize(current_state, p=2, dim=-1)
            
#             sim_execute = (norm_executed * target).sum(dim=-1)
#             sim_skip = (norm_current * target).sum(dim=-1)
            
#             # 3. 决定是否跳过
#             # 考虑额外奖励以鼓励跳过（如果有剩余跳层额度）
#             remaining_skips = self.max_skip_layers - sum(decisions)
#             if remaining_skips > 0:
#                 # 动态调整阈值：剩余跳层越多，越容易跳过
#                 dynamic_threshold = self.skip_threshold * (1.0 + 0.1 * remaining_skips)
#             else:
#                 dynamic_threshold = self.skip_threshold
                
#             do_skip = (sim_skip + dynamic_threshold) > sim_execute
            
#             # 如果已经跳过了太多层，强制执行
#             if sum(decisions) >= self.max_skip_layers:
#                 do_skip = False
            
#             # 4. 更新状态
#             if do_skip:
#                 # 跳过当前层，状态不变
#                 decisions.append(True)
#                 # current_state 保持不变
#             else:
#                 # 执行当前层
#                 decisions.append(False)
#                 current_state = executed_state
            
#             layer_similarities.append({
#                 'execute': sim_execute.item(),
#                 'skip': sim_skip.item(),
#                 'decision': do_skip.item()
#             })
        
#         # 收集跳过的层索引
#         skip_layers = [i for i, skip in enumerate(decisions) if skip]
        
#         return skip_layers, layer_similarities

#     def _forward_layer(self, layer_idx, hidden_state, past_key_value):
#         """单层前向传播"""
#         layer = self.model.model.layers[layer_idx]
        
#         # 准备输入
#         # hidden_state: [hidden_size]
#         h_seq = hidden_state.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
        
#         # Position IDs
#         past_len = past_key_value[0].shape[2]
#         position_ids = torch.tensor([[past_len]], dtype=torch.long, device=self.device)
        
#         # Attention mask
#         total_len = past_len + 1
#         mask = torch.zeros((1, 1, 1, total_len), device=self.device, dtype=self.dtype)
#         mask_val = torch.finfo(self.dtype).min
        
#         # 只能看到历史和自己，看不到其他候选
#         if total_len > past_len + 1:
#             mask[:, :, :, past_len+1:] = mask_val
        
#         # 前向传播
#         output = layer(
#             h_seq,
#             attention_mask=mask,
#             position_ids=position_ids,
#             past_key_value=past_key_value,
#             use_cache=False
#         )
        
#         return output[0].squeeze(0).squeeze(0)
    