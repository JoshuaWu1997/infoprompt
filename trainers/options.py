import argparse


def str2bool(v):
    return v.lower() in ('true', '1')


parser = argparse.ArgumentParser()

# -------------------------------------------------------------------------
# Data input settings
parser.add_argument("--root", type=str, default="data/", help="path to dataset")
parser.add_argument("--output_dir", type=str, default="./", help="output directory")
parser.add_argument("--task", type=str, default="mrpc", help="")
parser.add_argument("--split", type=str, default="train", help="")
# -------------------------------------------------------------------------
# Model input settings
parser.add_argument("--model_path", type=str, default="RoBertaPrompt", help="model name", )
# -------------------------------------------------------------------------
# Model params
parser.add_argument("--seed", type=int, default=42, help="only positive value enables a fixed seed")
# -------------------------------------------------------------------------
# Setting params
parser.add_argument("--n_prompt", type=int, default=1, help="number of soft prompts")
parser.add_argument("--n_latent", type=int, default=32, help="latent dimension")
parser.add_argument("--decay", type=float, default=0.00, help="decay rate for info-loss")
parser.add_argument("--beta", type=float, default=0.2, help="beta")
parser.add_argument("--gamma", type=float, default=0.1, help="gamma")
# -------------------------------------------------------------------------
# Optimization / training params
parser.add_argument('--continued', default=False, type=str2bool, help='Use GPU or CPU')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size (Adjust base on GPU memory)')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--epochs', default=30, type=int, help='Epochs')
# Other training environmnet settings
parser.add_argument('--cuda', default=True, type=str2bool, help='Use GPU or CPU')
parser.add_argument('--n_works', default=0, type=int, help='Number of worker threads in dataloader')
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Load Models
params, unparsed = parser.parse_known_args()
