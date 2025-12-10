import torch
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
class DynamicLayerOptimizer:
    """CLaSp 的动态层优化器"""
    
    def __init__(self, model, num_skip_layers):
        self.model = model
        self.num_skip_layers = num_skip_layers  # M
        self.num_layers = model.config.num_hidden_layers  # L
        self.hidden_size = model.config.hidden_size
        
    def optimize_skip_layers(self, last_hidden_states):
        """
        动态规划选择最优跳过层 (Algorithm 1)
        
        Args:
            last_hidden_states: [L+1, hidden_size] 上次验证的完整隐藏状态
                               (包含 embedding 层的输出)
        Returns:
            skip_layers: 最优跳过层列表
        """
        L = self.num_layers
        M = self.num_skip_layers
        d = self.hidden_size
        
        # g[i, j]: 跳过j层后第i层的最优隐藏状态
        g = torch.zeros(L + 1, M + 1, d, device=last_hidden_states.device,
                       dtype=last_hidden_states.dtype)
        g[0, 0] = last_hidden_states[0]
        
        # 记录决策路径
        decisions = torch.zeros(L + 1, M + 1, dtype=torch.bool, 
                               device=last_hidden_states.device)
        
        # 动态规划主循环
        for i in range(1, L + 1):
            g[i, 0] = last_hidden_states[i]
            l = min(i - 1, M)
            
            if l > 0:
                # 计算不跳过第i-1层的隐藏状态 (批量处理)
                G = self._forward_layer(i - 1, g[i - 1, 1:l + 1])
                
                # 计算 cosine 相似度
                F_cat = torch.cat([
                    F.normalize(G, dim=-1),
                    F.normalize(g[i - 1, :l], dim=-1)
                ], dim=0)
                
                target_norm = F.normalize(last_hidden_states[i].unsqueeze(0), dim=-1)
                sigma = (F_cat * target_norm).sum(dim=-1)
                
                sigma_no_skip = sigma[:l]
                sigma_skip = sigma[l:]
                
                # 决策：跳过还是不跳过
                max_no_skip = sigma_no_skip.max() if sigma_no_skip.numel() > 0 else -float('inf')
                max_skip = sigma_skip.max() if sigma_skip.numel() > 0 else -float('inf')

                if max_no_skip > max_skip:
                    g[i, 1:l + 1] = G
                    decisions[i, 1:l + 1] = False  # 不跳过
                else:
                    g[i, 1:l + 1] = g[i - 1, :l]
                    decisions[i, 1:l + 1] = True  # 跳过
                    
            if i <= M:
                g[i, i] = g[i - 1, i - 1]
                decisions[i, i] = True
           
        # 回溯找到最优跳过层集合
        skip_layers = self._backtrack(decisions, L, M)
        return skip_layers
    
    def _forward_layer(self, layer_idx, hidden_states):
        """
        前向传播单层 (批量处理多个候选状态)
        
        Args:
            layer_idx: 层索引
            hidden_states: (num_candidates, hidden_size) 多个候选隐藏状态
        Returns:
            (num_candidates, hidden_size) 前向传播后的隐藏状态
        """
        layer = self.model.model.layers[layer_idx]
        
        with torch.no_grad():
            target_dtype = layer.input_layernorm.weight.dtype
            hidden_states = hidden_states.to(target_dtype)
            
            num_candidates = hidden_states.shape[0]
            
            # 方法1: 逐个处理（更稳定）
            outputs = []
            for i in range(num_candidates):
                # 取出单个候选，形状 (hidden_size,)
                h = hidden_states[i:i+1]  # (1, hidden_size)
                
                # 扩展为 (batch=1, seq_len=1, hidden_size)
                h = h.unsqueeze(0)  # (1, 1, hidden_size)
                
                # 创建 position_ids (因为只有1个token，位置为0)
                position_ids = torch.zeros((1, 1), dtype=torch.long, device=h.device)
                
                # 调用 layer
                output = layer(
                    h,
                    position_ids=position_ids,
                    past_key_value=None,
                    use_cache=False
                )
                
                # output[0] 形状: (1, 1, hidden_size)
                # 压缩为 (1, hidden_size)
                outputs.append(output[0].squeeze(0))
            
            # 堆叠所有输出: (num_candidates, hidden_size)
            return torch.cat(outputs, dim=0)
    
    def _backtrack(self, decisions, L, M):
        """回溯找到最优跳过层"""
        skip_layers = []
        i, j = L, M
        
        while i > 0 and j > 0:
            if decisions[i, j]:
                skip_layers.append(i - 1)
                j -= 1
            i -= 1
        
        return sorted(skip_layers)