import torch.nn as nn
from hub.backbones import dinov2_vits14
from config import *
import torch.nn.functional as F
import vision_transformer
from deit import deit_model


def strip_prefix_from_state_dict(state_dict, prefix_to_strip="module.encoder_q."):
    return {k[len(prefix_to_strip):] if k.startswith(prefix_to_strip) else k: v
            for k, v in state_dict.items() if not k.startswith(prefix_to_strip + "head")}


class DINOBackbone(nn.Module):
    def __init__(self, backbone):
        super(DINOBackbone, self).__init__()

        self.backbone = backbone
        if self.backbone == 'dino':
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

        elif self.backbone == 'dinov2':
            self.model = dinov2_vits14()

        elif self.backbone == 'deit':
            self.model = deit_model

        else:
            raise NotImplementedError

    def forward(self, x):

        if self.backbone == 'dino':
            feat = self.model.forward_feats(x)
            features = feat[:, 1:]
            cls_token = feat[:, 0]

        elif self.backbone == 'dinov2':
            output = self.model.forward_features(x)
            feat = output['x_prenorm']
            features = feat[:, 1:]
            cls_token = feat[:, 0]

        elif self.backbone == 'deit':
            feat = self.model.forward_features(x, return_tokens=True)
            features = feat[:, 1:]
            cls_token = feat[:, 0]
        else:
            raise NotImplementedError

        n, p, c = features.shape
        features = torch.reshape(features, [n, int(p ** 0.5), int(p ** 0.5), c])
        features = features.permute([0, 3, 1, 2])
        features = features.contiguous()

        return cls_token, features


class TriHead(nn.Module):
    def __init__(self, args):
        """
        Three-channel output for separating:
          - Channel 0 -> Background
          - Channel 1 -> Foreground
          - Channel 2 -> Undefined
        """
        super(TriHead, self).__init__()
        self.output_channels = args.output_channels
        self.patch_classifier = nn.Linear(args.embedding_dim, args.num_classes)
        self.activation_head = nn.Conv2d(args.embedding_dim, self.output_channels, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(self.output_channels)
        self.dropout = nn.Dropout(args.drop_out_value_loc)


    def forward(self, x):
        """
        :param x: (B, C, N) feature map from backbone
        :return:
          logits_fg: (B, C)
          logits_am: (B, C)
          ccam:     (B, 3, H, W) probability map
        """

        N, C, H, W = x.shape

        # 1) Compute logits for each of the 3 categories at each pixel
        # seg_feats = torch.relu(self.seg_bn(self.seg_transform(x)))
        logits = self.bn_head(self.activation_head(x))  # (N, 3, H, W)

        # 2) Softmax across the channel dimension
        ccam = F.softmax(logits, dim=1)  # (N, 3, H, W)
        # ccam[:,0] => BG, ccam[:,1] => FG, ccam[:,2] => ambiguous

        # 3) Reshape ccam to (N, 3, H*W)
        ccam_flat = ccam.view(N, self.output_channels, H * W)

        # 4) Reshape x to (N, H*W, C) for matmul
        x_flat = x.view(N, C, H * W).permute(0, 2, 1)  # (N, H*W, C)

        # 5) For each category c, compute the average embedding
        # FG
        eps = 1e-6
        fg_map = ccam_flat[:, 0, :].unsqueeze(1)  # (N, 1, H*W)
        fg_sum = torch.sum(fg_map, dim=2, keepdim=True)  # (N, 1, 1)
        fg_feats = torch.matmul(fg_map, x_flat) / (fg_sum + eps)  # (N, 1, C)

        # BG
        bg_map = ccam_flat[:, 1, :].unsqueeze(1)  # (N, 1, H*W)
        bg_sum = torch.sum(bg_map, dim=2, keepdim=True)  # (N, 1, 1)
        bg_feats = torch.matmul(bg_map, x_flat) / (bg_sum + eps)  # (N, 1, C)

        fg_feats = fg_feats.reshape(x.size(0), -1)
        bg_feats = bg_feats.reshape(x.size(0), -1)

        fg_feats = self.dropout(fg_feats)
        bg_feats = self.dropout(bg_feats)

        logits_fg = self.patch_classifier(fg_feats)
        logits_bg = self.patch_classifier(bg_feats)

        # 6) Return
        return (logits,
                logits_fg,
                logits_bg,
                ccam[:, 0, :]
                )  # (N, H, W)

# Complete Model
class TriLite(nn.Module):
    def __init__(self, args):
        super(TriLite, self).__init__()
        self.backbone = DINOBackbone(args.backbone)
        self.triHead = TriHead(args)
        self.global_classifier = nn.Linear(args.embedding_dim, args.num_classes)
        self.dropout = nn.Dropout(args.drop_out_value_cls)

        if args.backbone == 'deit':
            head = self.backbone.model.head
            self.global_classifier.weight = head.weight
            self.global_classifier.bias = head.bias

    def forward(self, x):

        cls_token, patches = self.backbone(x)

        # localization
        pre_softmax_logits, logits_fg, logits_bg, localization_map = self.triHead(patches)

        # Classification
        cls_token = self.dropout(cls_token)
        logits = self.global_classifier(cls_token)

        return logits, pre_softmax_logits, logits_fg, logits_bg, localization_map


if __name__ == "__main__":
    import os
    import time
    from fvcore.nn import FlopCountAnalysis

    os.environ["XFORMERS_DISABLED"] = "1"

    args = create_arg_namespace("configs/CUB_config.yaml")

    x = torch.rand(1, 3, 224, 224).to(args.device)  # Add batch dimension
    model = TriLite(args).to(args.device)
    flops = FlopCountAnalysis(model, x).total()
    print(flops / 1e9, "GFLOPs")


    # warmup (important)
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)

    # timing
    iters = 100
    times = []

    with torch.no_grad():
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(x)

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    print(f"Avg inference time: {sum(times)/len(times):.2f} ms")

