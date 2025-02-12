from config import get_config
from trainer.trainer import Trainer
from utils.utils import setup_device

def main():
    config = get_config()
    device = setup_device(config.device)
    
    # Initialize trainer and start training/inference
    trainer = Trainer(config, device)
    trainer.train()
    
if __name__ == "__main__":
    main()
