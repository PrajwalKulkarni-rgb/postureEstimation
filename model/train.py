import torch
import datetime
import os
from torch.utils.data import DataLoader, Subset

# Modules
from utils.argparser import get_config
from utils.utils import set_seed, setup_logging
from nn.dataset import SkeletonDataset
from nn.model import STGCN
from nn.trainer import Trainer

def main():
    # 1. Config
    cfg = get_config()
    set_seed(cfg.SEED)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.RUN_NAME}_{timestamp}" if cfg.RUN_NAME else f"train_{timestamp}"  
    run_dir = os.path.join(cfg.RUNS_DIR, run_name)
    
    logger = setup_logging(run_dir)
    logger.info(f"--- Training Started: {run_name} ---")
    logger.info(f"Hyperparams: BS={cfg.BATCH_SIZE}, LR={cfg.LEARNING_RATE}, Dropout={cfg.DROPOUT}")

    # 2. Data
    try:
        full_dataset = SkeletonDataset(cfg.X_PATH, cfg.Y_PATH, seq_length=cfg.SEQ_LENGTH, augment=cfg.AUGMENT)
    except FileNotFoundError as e:
        logger.error(e)
        return

    n = len(full_dataset)
    train_size = int(cfg.TRAIN_SPLIT * n)

    # Seeded deterministic split — separate generator so it doesn't disturb other random state
    generator = torch.Generator().manual_seed(cfg.SEED)
    indices = torch.randperm(n, generator=generator).tolist()
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    # IMPORTANT: val and train must use SEPARATE dataset objects.
    # random_split shares one object, so val_data.dataset.augment = False
    # would silently disable augmentation for training too.
    val_dataset = SkeletonDataset(cfg.X_PATH, cfg.Y_PATH, seq_length=cfg.SEQ_LENGTH, augment=False)

    train_data = Subset(full_dataset, train_idx)  # full_dataset has augment=cfg.AUGMENT
    val_data   = Subset(val_dataset,  val_idx)    # val_dataset has augment=False

    train_loader = DataLoader(train_data, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    # 3. Model (WITH DROPOUT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STGCN(
        num_classes=cfg.NUM_CLASSES, 
        in_channels=cfg.IN_CHANNELS,
        dropout=cfg.DROPOUT  # <--- Critical Update
    ).to(device)
    
    logger.info(f"Model initialized on {device}.")

    # 4. Train
    trainer = Trainer(model, train_loader, val_loader, cfg, logger, device)
    trainer.fit(run_dir)
    
    logger.info(f"Done. Best Accuracy: {trainer.best_acc:.2f}%")

if __name__ == "__main__":
    main()