import torch
import torch.nn.functional as F

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
            last_hidden_states: [L, hidden_size] 上次验证的完整隐藏状态
        Returns:
            attn_skip_layers, mlp_skip_layers: 最优跳过层集合
        """
        L = self.num_layers
        M = self.num_skip_layers
        d = self.hidden_size
        
        # g[i, j]: 跳过j层后第i层的最优隐藏状态
        g = torch.zeros(L + 1, M + 1, d, device=last_hidden_states.device)
        g[0, 0] = last_hidden_states[0]
        
        # 记录决策路径
        decisions = torch.zeros(L + 1, M + 1, dtype=torch.bool, device=last_hidden_states.device)
        
        # 动态规划主循环
        for i in range(1, L + 1):
            g[i, 0] = last_hidden_states[i]
            l = min(i - 1, M)
            
            if l > 0:
                # 计算不跳过第i-1层的隐藏状态
                with torch.no_grad():
                    G = self._forward_layer(i - 1, g[i - 1, 1:l + 1])
                
                # 计算 cosine 相似度
                F_cat = torch.cat([
                    F.normalize(G, dim=-1),
                    F.normalize(g[i - 1, :l], dim=-1)
                ], dim=0)
                
                target_norm = F.normalize(last_hidden_states[i].unsqueeze(0), dim=-1)
                #sigma = torch.matmul(F_cat, target_norm.T).squeeze()
                sigma = (F_cat * target_norm).sum(dim=-1).squeeze()
                # 插入调试代码：
                # print(f"DEBUG: Layer i={i}, Max skips M={M}, Current l={l}")
                # print(f"DEBUG: sigma shape: {sigma.shape}, numel: {sigma.numel()}")
                sigma_no_skip = sigma[:l] # 对应 F_cat 的上半部分 (No Skip)
                sigma_skip = sigma[l:]    # 对应 F_cat 的下半部分 (Skip)
                # print(f"DEBUG: sigma[:l] (No Skip) shape: {sigma_no_skip.shape}, numel: {sigma_no_skip.numel()}")
                # print(f"DEBUG: sigma[l:] (Skip) shape: {sigma_skip.shape}, numel: {sigma_skip.numel()}")
                if sigma_no_skip.numel() == 0 or sigma_skip.numel() == 0:
                    print(f"CRITICAL ALERT: Empty tensor detected! i={i}, l={l}")
                # 决策：跳过还是不跳过
                # 添加防御性检查：如果切片为空，则最大值设为负无穷大
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
        """前向传播单层（简化版本）"""
        layer = self.model.model.layers[layer_idx]
        
        # 这里需要根据实际情况调整
        with torch.no_grad():
            target_dtype = layer.input_layernorm.weight.dtype 
            hidden_states = hidden_states.to(target_dtype)
            output = layer(hidden_states.unsqueeze(0))
            return output[0].squeeze(0)
    
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