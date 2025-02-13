class Config:
    # General settings
    project_name = "MoASE-Mac"  
    num_classes = 10        # Number of output classes

    # Data settings
    batch_size = 32         # Batch size for training and validation
    num_workers = 4         # Number of workers for DataLoader

    # Training settings
    max_epochs = 10         # Number of epochs to train
    learning_rate = 1e-3    # Learning rate for the optimizer

    # Loss configuration
    task_loss_type = 'jsd'  # Options: 'jsd', 'label_smoothing', or 'ce'
    num_splits = 3          # Number of splits for JSD loss
    alpha = 12              # Scaling factor for KL divergence in JSD loss
    smoothing = 0.1         # Label smoothing factor
    lambda_hp = 0.1         # Weight for Homeostatic-Proximal loss
    lambda_cons = 0.05      # Weight for consistency loss

    # Hardware settings
    use_mps = True          # Use MPS (Metal Performance Shaders) for Mac with M1/M2 chip
