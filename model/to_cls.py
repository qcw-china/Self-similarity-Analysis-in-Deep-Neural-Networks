import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class CLSTokenAggregator(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes):
        super(CLSTokenAggregator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # 初始化CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # 为每一层定义独立的注意力机制参数
        self.layer_attentions = nn.ModuleList([
            nn.ModuleDict({
                'query': nn.Linear(hidden_size, hidden_size),
                'key': nn.Linear(hidden_size, hidden_size),
                'value': nn.Linear(hidden_size, hidden_size),
            })
            for _ in range(num_layers)
        ])

        # 分类头
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, layer_features):
        # layer_features: list of tensors, each of shape (batch_size, seq_len, hidden_size)
        batch_size = layer_features[0].size(0)

        # 将CLS Token扩展到batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_size)

        # 逐层聚合特征
        for i in range(self.num_layers):
            # 获取当前层的特征
            current_features = layer_features[i]  # (batch_size, seq_len, hidden_size)

            # 将CLS Token与当前层的特征拼接
            combined_features = torch.cat([cls_tokens, current_features], dim=1)  # (batch_size, seq_len + 1, hidden_size)

            # 使用当前层的独立注意力参数
            attention_layer = self.layer_attentions[i]
            q = attention_layer['query'](cls_tokens)  # (batch_size, 1, hidden_size)
            k = attention_layer['key'](combined_features)  # (batch_size, seq_len + 1, hidden_size)
            v = attention_layer['value'](combined_features)  # (batch_size, seq_len + 1, hidden_size)

            # 计算注意力得分
            attention_scores = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, 1, seq_len + 1)
            attention_scores = attention_scores / (self.hidden_size ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, 1, seq_len + 1)

            # 加权求和
            weighted_features = torch.matmul(attention_weights, v)  # (batch_size, 1, hidden_size)

            # 使用残差连接更新 CLS Token
            cls_tokens = cls_tokens + weighted_features  # 残差连接

        # 最终的 CLS Token 特征
        final_cls_token = cls_tokens.squeeze(1)  # (batch_size, hidden_size)

        # 通过分类头
        logits = self.classifier(final_cls_token)  # (batch_size, num_classes)

        return logits


if __name__ == '__main__':
    # 示例用法
    batch_size = 8
    seq_len = 10
    hidden_size = 128
    num_layers = 6
    num_classes = 10

    # 假设我们有6层的隐层特征
    layer_features = [torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers)]

    # 初始化模块
    model = CLSTokenAggregator(hidden_size, num_layers, num_classes)

    # 前向传播
    logits = model(layer_features)
    print(logits.shape)  # 输出: torch.Size([8, 10])