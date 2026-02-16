from data_loaders import get_data_loader
from model import TriLite
from train import evaluate_w_localization
from config import create_arg_namespace
import torch

def evaluate(args):
    triLite = TriLite(args)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    triLite.load_state_dict(checkpoint['state_dict'], strict=False)
    triLite = triLite.to(args.device)

    data_loaders = get_data_loader(args,
                                   args.data_roots,
                                   args.metadata_root,
                                   args.batch_size,
                                   args.workers,
                                   args.resize_size,
                                   args.crop_size,
                                   args.resize_eval)

    loc_score, classification_acc, top_k_loc_accuracies, _ = evaluate_w_localization(triLite,
                                                                                     data_loaders['test'],
                                                                                     0,
                                                                                     args,
                                                                                     split="test")

    if args.dataset_name == "OpenImages":

        print(f'localizationV3/test: {loc_score}, cls acc: {classification_acc}')
    else:
        print(f'localizationV3/test: {loc_score}, '
              f'cls acc: {classification_acc}, '
              f'top_1_acc: {top_k_loc_accuracies["TOP_1"]},'
              f' top_5_acc: {top_k_loc_accuracies["TOP_5"]}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run WSOL with a YAML config")

    parser.add_argument(
        '--config',
        type=str,
        default=f'configs/CUB_config.yaml',
        help='Path to YAML config file (default: configs/CUB_config.yaml)'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=f'path_to_checkpoint.pth',
        help='Path to model checkpoint (e.g., checkpoint/CUB/best_combined.pth)'
    )

    cli_args = parser.parse_args()
    args = create_arg_namespace(cli_args.config)
    args.checkpoint = cli_args.checkpoint  # add it to the namespace
    evaluate(args)
