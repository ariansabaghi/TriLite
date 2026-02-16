import torch
from tqdm import tqdm
from config import *
from inference import CAMComputer
import os
from tb_vis import generate_overlay_heatmap
import torch.nn.functional as F
import time


def multi_task_loss(
    images,         # (N, 3, 224, 224) input image tensors
    labels,        # (N,) ground-truth class indices
    model,    # final classification head: maps (N, C) -> (N, num_classes)
    alpha=0,
    eps=1e-7,
):
    """
    Implements:
      L = L_cls + L_fg + alpha * L_am

    :param feats:        output of your ViT/backbone, shape (N, C, H, W)
    :param labels:       ground-truth classes, shape (N,)
    :param model:        TriLite model
    :param alpha:        weight for FG classification loss
    :param eps:          small constant for numerical stability
    :return: total_loss, dict of individual losses
    """

    N = images.shape[0]

    # ------------------------------------------------------------------------
    # 1)CLASSIFICATION Loss
    # ------------------------------------------------------------------------
    logits, pre_softmax_logits, logits_fg, logits_bg, localization_map = model(images)
    cls_loss = F.cross_entropy(logits, labels)
    # ------------------------------------------------------------------------
    # 2) FOREGROUND Loss
    # ------------------------------------------------------------------------
    loss_fg = F.cross_entropy(logits_fg, labels)

    # ------------------------------------------------------------------------
    # 3) Adversarial Loss
    # ------------------------------------------------------------------------
    # We want to penalize the model if the BG vector alone can classify the image correctly.
    # Probability that BG is correct for each sample
    # p_bg_correct = p(class=labels[i] | BG)
    probs_bg = F.softmax(logits_bg, dim=1)
    p_bg_correct = probs_bg[torch.arange(N), labels]  # shape (N,)
    # We penalize -log(1 - p_am_correct)
    # i.e. if UF can classify well, we get a big penalty
    loss_bg = -torch.log(1.0 - p_bg_correct + eps).mean() * alpha

    total_loss = loss_fg + loss_bg + cls_loss

    # Optionally return a dict of the terms for logging
    loss_dict = {
        'cls_loss': cls_loss.item(),
        'loss_bg': loss_bg.item(),
        'loss_fg':   loss_fg.item(),
        'total':     total_loss.item()
    }
    return total_loss, loss_dict, logits


def train(model, train_loader, optimizer, epoch, args):
    model.train()

    total_loss_dict = {}
    num_correct = 0
    num_images = 0

    # -------- efficiency accumulators --------
    total_compute_time = 0.0   # seconds (forward + backward + step)
    total_images = 0
    peak_mem_mb_epoch = 0.0

    # ensure clean memory stats at epoch start
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    end = time.perf_counter()

    for images, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
        images, targets = images.to(args.device), targets.to(args.device)

        # ---- COMPUTE TIMING START ----
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        loss, loss_dict, logits = multi_task_loss(images, targets, model, alpha=args.alpha)

        for k, v in loss_dict.items():
            if k not in total_loss_dict:
                total_loss_dict[k] = v
            else:
                total_loss_dict[k] += v


        preds = logits.argmax(dim=1)
        num_correct += (preds == targets).sum().item()

        num_images += images.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        # ---- COMPUTE TIMING END ----

        compute_time = t1 - t0
        total_compute_time += compute_time
        bs = images.size(0)
        total_images += bs

        # track peak memory (MB)
        if torch.cuda.is_available():
            peak_mem_mb_epoch = max(
                peak_mem_mb_epoch,
                torch.cuda.max_memory_allocated() / (1024 ** 2)
            )

    for k, v in total_loss_dict.items():
        total_loss_dict[k] = v / len(train_loader)
    classification_acc = num_correct / float(num_images) * 100


    sec_per_img = total_compute_time / total_images
    img_per_sec = total_images / total_compute_time

    # -------- log to stdout (goes to your log file) --------
    print(
        f"[EFF][Epoch {epoch}] "
        f"sec/img={sec_per_img:.4f} | "
        f"img/s={img_per_sec:.1f} | "
        f"peakMem={peak_mem_mb_epoch:.0f}MB | "
        f"images={total_images}"
    )

    return total_loss_dict, classification_acc


def evaluate_w_localization(model, data_loader, epoch, args, split="val", summary_writer=None):
    model.eval()

    num_correct = 0
    num_images = 0
    total_loss_dict = {}

    if summary_writer is not None:
        overlays = generate_overlay_heatmap(model, args)
        # Add images to the summary writer
        for i in range(len(overlays)):
            summary_writer.add_image("fg_map_{}".format(i), overlays[i], global_step=epoch)

    for images, targets, _ in tqdm(data_loader, desc=f"Epoch {epoch}"):
        images, targets = images.to(args.device), targets.to(args.device)

        with torch.no_grad():
            loss, loss_dict, logits = multi_task_loss(images, targets, model, args.alpha)

            for k, v in loss_dict.items():
                if k not in total_loss_dict:
                    total_loss_dict[k] = 0
                else:
                    total_loss_dict[k] += v


        preds = logits.argmax(dim=1)
        num_correct += (preds == targets).sum().item()
        num_images += images.size(0)

    for k, v in total_loss_dict.items():
        total_loss_dict[k] = v / len(data_loader)

    classification_acc = num_correct / float(num_images) * 100


    cam_computer = CAMComputer(
        model=model,
        loader=data_loader,
        metadata_root=os.path.join(args.metadata_root, split),
        mask_root=getattr(args, "mask_root", None),
        iou_threshold_list=args.iou_threshold_list,
        dataset_name=args.dataset_name,
        split=split,
        cam_curve_interval=args.cam_curve_interval,
        multi_contour_eval=args.multi_contour_eval,
    )

    cam_performance, top_k_loc_accuracies = cam_computer.compute_and_evaluate_cams()
    if args.dataset_name != "OpenImages":
        loc_score = cam_performance[args.iou_threshold_list.index(50)]
    else:
        loc_score = cam_performance

    if split == "val":
        return loc_score, classification_acc, top_k_loc_accuracies, total_loss_dict

    else:
        return cam_performance, classification_acc, top_k_loc_accuracies, None

