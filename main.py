import torch
import pytorch_lightning as pl
from trainer.trainer import MoASETrainer
from config import Config

def main():
    # Load configuration
    config = Config()

    # Check if MPS is available for Mac
    if config.use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for training.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device.type} device for training.")

    # Initialize the Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    # Create the model with the given config
    model = MoASETrainer(config)

    # Start training
    trainer.fit(model)

if __name__ == "__main__":
    main()
