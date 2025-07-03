import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import torch
import torch.nn as nn
from common.init import xavier_uniform_initialization


class EmbeddingWrapper(nn.Module):
    def __init__(self, user_emb_tensor, item_emb_tensor, strategy="delta", r=None):
        super().__init__()
        self.strategy = strategy
        if strategy == "direct":
            self.trainable = True
        elif strategy in ["delta", "lora"]:
            self.trainable = False
        else:
            raise NotImplementedError(f"Strategy {strategy} is not implemented.")

        self.pretrained_user_emb = nn.Parameter(user_emb_tensor, requires_grad=self.trainable)
        self.pretrained_item_emb = nn.Parameter(item_emb_tensor, requires_grad=self.trainable)

        if not self.trainable:
            num_users, dim = user_emb_tensor.size()
            num_items, _ = item_emb_tensor.size()
            if strategy == "delta":
                self.delta_user_emb = nn.Embedding(num_users, dim)
                self.delta_item_emb = nn.Embedding(num_items, dim)
                xavier_uniform_initialization(self.delta_user_emb)
                xavier_uniform_initialization(self.delta_item_emb)

            elif strategy == "lora":
                self.a_user_emb = nn.Embedding(num_users, r)
                self.b_user_emb = nn.Parameter(torch.empty(r, dim))
                self.a_item_emb = nn.Embedding(num_items, r)
                self.b_item_emb = nn.Parameter(torch.empty(r, dim))
                xavier_uniform_initialization(self.a_user_emb)
                xavier_uniform_initialization(self.a_item_emb)
                xavier_uniform_initialization(self.b_user_emb)
                xavier_uniform_initialization(self.b_item_emb)

    def forward(self):
        if self.strategy == "direct":
            return self.pretrained_user_emb, self.pretrained_item_emb
        elif self.strategy == "delta":
            return (
                self.pretrained_user_emb + self.delta_user_emb.weight,
                self.pretrained_item_emb + self.delta_item_emb.weight,
            )
        elif self.strategy == "lora":
            user_emb = self.pretrained_user_emb + self.a_user_emb.weight @ self.b_user_emb
            item_emb = self.pretrained_item_emb + self.a_item_emb.weight @ self.b_item_emb
            return user_emb, item_emb

    def freeze_pretrained(self):
        self.pretrained_user_emb.requires_grad_(False)
        self.pretrained_item_emb.requires_grad_(False)

    def unfreeze_pretrained(self):
        self.pretrained_user_emb.requires_grad_(True)
        self.pretrained_item_emb.requires_grad_(True)

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
