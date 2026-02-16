from config import *
from data_loaders import get_data_loader
from model import TriLite

from train import train, evaluate_w_localization
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from util import set_seed


def main(args):
    set_seed(42)

    checkpoints_dir = os.path.join("checkpoint", args.dataset_name, args.experiment_name)

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir, exist_ok=True)

    writer = SummaryWriter(os.path.join(args.log_dir, args.experiment_name))


    data_loaders = get_data_loader(args,
                                   args.data_roots,
                                   args.metadata_root,
                                   args.batch_size,
                                   args.workers,
                                   args.resize_size,
                                   args.crop_size,
                                   args.resize_eval)


    triLite = TriLite(args)
    triLite = triLite.to(args.device)


    # only making classifier trainable
    for name, param in triLite.named_parameters():
        if 'global_classifier' in name or 'triHead' in name:
        # if not 'global_classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    total_params = sum(p.numel() for p in triLite.parameters())
    trainable_params = sum(p.numel() for p in triLite.parameters() if p.requires_grad)

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")


    text = '<br>'
    all_params = 0
    trainable_params = 0
    for name, param in triLite.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        text += f"{name}: {param.requires_grad}<br>"


    writer.add_text('Params',
                    text + f'<br> All Params: {all_params}<br>Trainable Params: {trainable_params}',
                    0)

    param_groups = [
        {"params": [param for name, param in triLite.named_parameters() if "global_classifier" in name],
         "lr": args.lr*args.lr_multiplier, "weight_decay": args.weight_decay},
        {"params": [param for name, param in triLite.named_parameters() if "triHead" in name],
         "lr": args.lr, "weight_decay": args.weight_decay},
    ]

    optimizer = torch.optim.AdamW(params=param_groups)

    best_val_acc = 0
    best_loc_score = 0

    best_loc_epoch = 0
    best_cls_epoch = 0

    patience_counter = 0


    for epoch in tqdm(range(args.epochs)):
        epoch += 1
        train_loss, train_cls_acc = train(triLite, data_loaders['train'], optimizer, epoch, args)

        loc_score, val_cls_acc, top_k_loc_accuracies, val_loss = evaluate_w_localization(triLite, data_loaders['val'], epoch,
                                                                           args, split="val", summary_writer=writer)

        # Print the learning rate for demonstration purposes
        for param_group in optimizer.param_groups:
            print(f'Epoch {epoch}, LR: {param_group["lr"]}')

        save_path = os.path.join(checkpoints_dir, "best_combined.pth")

        # Load current state dict
        current_state_dict = triLite.state_dict()

        # We use a persistent dictionary to store best weights
        if val_cls_acc > best_val_acc:
            patience_counter = 0            # we just consider localization performance to reset the patience value
            best_val_acc = val_cls_acc
            print("Updating classifier head in checkpoint...")

            if os.path.exists(save_path):
                state_dict = torch.load(save_path, map_location=args.device)['state_dict']
            else:
                state_dict = current_state_dict.copy()

            for name in current_state_dict:
                if 'global_classifier' in name:
                    state_dict[name] = current_state_dict[name]

            torch.save({
                "state_dict": state_dict,
                "best_cls_epoch": best_cls_epoch,
                "best_loc_epoch": best_loc_epoch
            }, save_path)

        if loc_score > best_loc_score:
            patience_counter = 0
            best_loc_score = loc_score
            best_loc_epoch = epoch
            print("Updating localization head in checkpoint...")

            if os.path.exists(save_path):
                state_dict = torch.load(save_path, map_location=args.device)['state_dict']
            else:
                state_dict = current_state_dict.copy()

            for name in current_state_dict:
                if 'triHead' in name:
                    state_dict[name] = current_state_dict[name]

            torch.save({
                "state_dict": state_dict,
                "best_cls_epoch": best_cls_epoch,
                "best_loc_epoch": best_loc_epoch
            }, save_path)

        else:
            patience_counter += 1

        writer.add_scalar('loc_score/val', loc_score, epoch)
        writer.add_scalar('Accuracy/val', val_cls_acc, epoch)

        for k, v in val_loss.items():
            writer.add_scalar(f'{k}/val', v, epoch)

        for k, v in train_loss.items():
            writer.add_scalar(f'{k}/train', v, epoch)


        if patience_counter == args.early_stopping_patience:
            print("Early stopping triggered")
            break


    # evaluation on test set based on the highest localization acc on validation set
    checkpoint = torch.load(os.path.join(checkpoints_dir, "best_combined.pth"), map_location=args.device)
    triLite.load_state_dict(checkpoint["state_dict"])
    loc_score, classification_acc, top_k_loc_accuracies, _ = evaluate_w_localization(triLite,
                                                                                     data_loaders['test'],
                                                                                     0,
                                                                                     args,
                                                                                     split="test")
    if args.dataset_name == "OpenImages":
        writer.add_text('performance (test)',
                        f'localizationV3/test: {loc_score},'
                        f' cls acc: {classification_acc},',
                        0)
        print(f'localizationV3/test: {loc_score}, cls acc: {classification_acc}')

    else:
        writer.add_text('performance (test)',
                        f'localizationV3/test: {loc_score},'
                        f' cls acc: {classification_acc},'
                        f' top_1_acc: {top_k_loc_accuracies["TOP_1"]},'
                        f' top_5_acc: {top_k_loc_accuracies["TOP_5"]}',
                        0)
        print(f'localizationV3/test: {loc_score}, cls acc: {classification_acc}, top_1_acc: {top_k_loc_accuracies["TOP_1"]}, top_5_acc: {top_k_loc_accuracies["TOP_5"]}')

    writer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run WSOL with a YAML config")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/OpenImages_config.yaml',
        help='Path to YAML config file (default: configs/CUB_config.yaml)'
    )
    cli_args = parser.parse_args()

    args = create_arg_namespace(cli_args.config)
    main(args)
