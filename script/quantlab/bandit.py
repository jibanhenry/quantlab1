# -*- coding: utf-8 -*-
# 预留：上下文 bandit（策略×桶）在线权重分配；此处给出简单占位实现
class SimpleBandit:
    def __init__(self, strategies=('S1','S2','S3')):
        self.weights = {s: 1.0 for s in strategies}
    def update(self, reward_by_strategy):
        for s, r in reward_by_strategy.items():
            self.weights[s] = max(0.1, self.weights.get(s,1.0) + r)
    def get_weights(self):
        total = sum(self.weights.values())
        return {k: v/total for k,v in self.weights.items()} if total>0 else self.weights
