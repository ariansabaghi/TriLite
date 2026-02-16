import yaml
from argparse import Namespace
import torch

def load_yaml_config(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def create_arg_namespace(yaml_path: str) -> Namespace:
    config = load_yaml_config(yaml_path)
    config["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return Namespace(**config)
