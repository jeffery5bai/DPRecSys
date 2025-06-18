import torch.nn as nn


class EmbeddingWrapper(nn.Module):
    def __init__(self, user_emb_tensor, item_emb_tensor, trainable=True):
        super().__init__()
        self.trainable = trainable

        self.pretrained_user_emb = nn.Parameter(user_emb_tensor, requires_grad=trainable)
        self.pretrained_item_emb = nn.Parameter(item_emb_tensor, requires_grad=trainable)

        if not trainable:
            num_users, dim = user_emb_tensor.size()
            num_items, _ = item_emb_tensor.size()
            self.delta_user_emb = nn.Embedding(num_users, dim)
            self.delta_item_emb = nn.Embedding(num_items, dim)

    def forward(self):
        if self.trainable:
            return self.pretrained_user_emb, self.pretrained_item_emb
        else:
            return (
                self.pretrained_user_emb + self.delta_user_emb.weight,
                self.pretrained_item_emb + self.delta_item_emb.weight,
            )

    def freeze_pretrained(self):
        self.pretrained_user_emb.requires_grad_(False)
        self.pretrained_item_emb.requires_grad_(False)

    def unfreeze_pretrained(self):
        self.pretrained_user_emb.requires_grad_(True)
        self.pretrained_item_emb.requires_grad_(True)
