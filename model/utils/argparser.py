import argparse
import os
from dataclasses import dataclass, field
from typing import List

# --- 1. OPTUNA WINNING SETTINGS ---
DEFAULTS = {
    # Paths
    "DATA_DIR": "Data/",
    "X_PATH": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trainable_data/x_train.npy"),
    "Y_PATH": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trainable_data/y_train.npy"),
    "RUNS_DIR": "runs",
    "MODEL_SAVE_NAME": "stgcn_posture_model.pth",
    
    # Best Hyperparameters (from Optuna)
    "BATCH_SIZE": 64,           # Optuna found 64
    "EPOCHS": 100,              # Running longer for max accuracy
    "LR": 0.008,                # Optuna found ~0.0079
    "WEIGHT_DECAY": 1.5e-5,     # Optuna found ~1.49e-5
    "DROPOUT": 0.16,            # Optuna found ~0.157
    "LABEL_SMOOTHING": 0.02,    # Optuna found ~0.019
    
    # Scheduler
    "LR_MILESTONES": [40, 80],
    "LR_GAMMA": 0.1,
    "PATIENCE": 20,
    
    # Model/Data
    "NUM_CLASSES": 3,
    "IN_CHANNELS": 6, # 3 Pos + 3 Vel
    "SEQ_LENGTH": 50,
    "TRAIN_SPLIT": 0.8,
    "SEED": 42
}

@dataclass
class Config:
    """Runtime configuration object."""
    DATA_DIR: str
    X_PATH: str
    Y_PATH: str
    RUNS_DIR: str
    MODEL_SAVE_NAME: str
    
    BATCH_SIZE: int
    EPOCHS: int
    LEARNING_RATE: float
    WEIGHT_DECAY: float
    LR_MILESTONES: List[int]
    LR_GAMMA: float
    LABEL_SMOOTHING: float
    DROPOUT: float  # <--- Added this
    EARLY_STOPPING_PATIENCE: int
    
    NUM_CLASSES: int
    IN_CHANNELS: int
    SEQ_LENGTH: int
    TRAIN_SPLIT: float
    SEED: int
    AUGMENT: bool = True
    RUN_NAME: str = None

def get_config():
    parser = argparse.ArgumentParser(
        description="Industrial Safety ST-GCN Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment
    exp = parser.add_argument_group('Experiment')
    exp.add_argument("--name", type=str, default=None, help="Run name")
    exp.add_argument("--seed", type=int, default=DEFAULTS["SEED"], help="Random seed")
    exp.add_argument("--runs_dir", type=str, default=DEFAULTS["RUNS_DIR"], help="Output dir")

    # Hyperparameters
    hparam = parser.add_argument_group('Hyperparameters')
    hparam.add_argument("--batch_size", type=int, default=DEFAULTS["BATCH_SIZE"], help="Batch size")
    hparam.add_argument("--epochs", type=int, default=DEFAULTS["EPOCHS"], help="Epochs")
    hparam.add_argument("--lr", type=float, default=DEFAULTS["LR"], help="Learning rate")
    hparam.add_argument("--weight_decay", type=float, default=DEFAULTS["WEIGHT_DECAY"], help="Weight decay")
    hparam.add_argument("--dropout", type=float, default=DEFAULTS["DROPOUT"], help="Dropout rate") # Added CLI arg
    hparam.add_argument("--label_smoothing", type=float, default=DEFAULTS["LABEL_SMOOTHING"], help="Label smoothing")
    hparam.add_argument("--patience", type=int, default=DEFAULTS["PATIENCE"], help="Early stopping patience")
    
    # Scheduler
    hparam.add_argument("--lr_gamma", type=float, default=DEFAULTS["LR_GAMMA"], help="LR decay factor")
    hparam.add_argument("--lr_step", type=int, nargs='+', default=DEFAULTS["LR_MILESTONES"], help="Decay steps")

    # Data
    arch = parser.add_argument_group('Model')
    arch.add_argument("--seq_len", type=int, default=DEFAULTS["SEQ_LENGTH"], help="Sequence length")
    arch.add_argument("--in_channels", type=int, default=DEFAULTS["IN_CHANNELS"], help="Input channels")
    arch.add_argument("--no_augment", action="store_true", help="Disable augmentation")

    args = parser.parse_args()

    cfg = Config(
        DATA_DIR=DEFAULTS["DATA_DIR"],
        X_PATH=DEFAULTS["X_PATH"],
        Y_PATH=DEFAULTS["Y_PATH"],
        RUNS_DIR=args.runs_dir,
        MODEL_SAVE_NAME=DEFAULTS["MODEL_SAVE_NAME"],
        
        BATCH_SIZE=args.batch_size,
        EPOCHS=args.epochs,
        LEARNING_RATE=args.lr,
        WEIGHT_DECAY=args.weight_decay,
        DROPOUT=args.dropout, # Pass it through
        LR_MILESTONES=args.lr_step,
        LR_GAMMA=args.lr_gamma,
        LABEL_SMOOTHING=args.label_smoothing,
        EARLY_STOPPING_PATIENCE=args.patience,
        
        NUM_CLASSES=DEFAULTS["NUM_CLASSES"],
        IN_CHANNELS=args.in_channels,
        SEQ_LENGTH=args.seq_len,
        TRAIN_SPLIT=DEFAULTS["TRAIN_SPLIT"],
        SEED=args.seed,
        
        AUGMENT=not args.no_augment,
        RUN_NAME=args.name
    )
    
    return cfg