import argparse

def get_config():
    parser = argparse.ArgumentParser(description="MoASE-Mac Configurations")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="mps", help="Device to use: mps, cuda, or cpu")
    # Add more configuration options as needed
    return parser.parse_args()
