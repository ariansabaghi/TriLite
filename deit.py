import types
import torch
import timm

# timm originally does not return patch tokens so we modify the original forward_features method
def forward_features_patched(self, x, return_tokens: bool = False):
    x = self.patch_embed(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)

    if getattr(self, "dist_token", None) is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

    x = self.pos_drop(x + self.pos_embed)
    x = self.blocks(x)
    x = self.norm(x)

    if return_tokens:
        return x  # (B, N, D)

    if getattr(self, "dist_token", None) is None:
        return self.pre_logits(x[:, 0])
    else:
        return x[:, 0], x[:, 1]

# ---- create model normally ----
deit_model = timm.create_model("deit_small_patch16_224", pretrained=True)

# ---- patch ONLY this instance ----
deit_model.forward_features = types.MethodType(forward_features_patched, deit_model)

if __name__ == '__main__':

    x = torch.randn(2, 3, 224, 224)
    tokens = deit_model.forward_features(x, return_tokens=True)
    cls_feat = tokens[:, 0]
    patch_feats = tokens[:, 1:]