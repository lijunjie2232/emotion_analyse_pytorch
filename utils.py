import torch
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, get_rank
import os
import argparse
import logging
import sys
from pathlib import Path
from matplotlib import pyplot as plt


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    scaler=None,
    device="cuda",
    fp16=True,
    progress=True,
):
    model.train()
    loop = tqdm(train_loader, desc="train", leave=False) if progress else train_loader
    total_loss = 0.0
    total_count = 0
    ateru = 0
    assert scaler is not None or not fp16

    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(
            device_type=device,
            enabled=fp16,
            dtype=torch.float16 if fp16 else torch.float32,
        ):
            output = model(data)
            loss = criterion(output, target)
        if fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_count += len(data)
        # calc acc
        ateru += (output.argmax(dim=1) == target).sum().item()
        total_loss += loss.item()
        if progress:
            loop.set_postfix(
                {
                    "acc": ateru / total_count,
                    "loss": total_loss / (batch_idx + 1),
                }
            )
    scheduler.step()
    return ateru / total_count, total_loss / (batch_idx + 1)


@torch.no_grad()
def val_epoch(
    model,
    val_loader,
    criterion,
    device="cuda",
    progress=True,
):
    model.eval()
    loop = tqdm(val_loader, desc="val", leave=False) if progress else val_loader
    val_loss = 0.0
    total_num = 0
    total_correct = 0
    for i, (inputs, labels) in enumerate(loop):
        inputs = inputs.to(device)
        labels = labels.to(device)
        logist, _ = model(inputs)
        loss = criterion(logist, labels)
        val_loss += loss.item() * inputs.size(0)
        total_num += labels.size(0)
        total_correct += (logist.argmax(dim=1) == labels).sum().item()
        if progress:
            loop.set_postfix(loss=val_loss / (i + 1), acc=total_correct / total_num)
    val_loss = val_loss / len(val_loader.dataset)
    return total_correct / total_num, val_loss / (i + 1)


def save(model, optimizer, epoch, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load(model, optimizer, path):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return epoch


def ddp_setup(
    rank: int,
    world_size: int,
    backend="nccl",
    master_addr="localhost",
    master_port=12355,
):
    """
    Args:
        rank: Unique identifier of each process
    world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    init_process_group(
        backend=backend, rank=rank, world_size=world_size, init_method="env://"
    )


def ddp_cleanup():
    destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training configuration for facial expression dataset",
    )

    # Training parameters
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="Step size for learning rate scheduler (default: 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        help="Number of epochs to train (default: 120)",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=-1,
        help="Epoch to start training from, use -1 to start from scratch (default: -1)",
    )
    parser.add_argument(
        "--train_patience",
        type=int,
        default=20,
        help="Patience for early stopping (default: 20)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed precision training",
    )

    # Checkpoint paths
    parser.add_argument(
        "--best_checkpoint",
        type=str,
        default="best_model.pt",
        help="Path to save the best model checkpoint (default: best_model.pt)",
    )
    parser.add_argument(
        "--last_checkpoint",
        type=str,
        default="last_model.pt",
        help="Path to save the last model checkpoint (default: last_model.pt)",
    )

    # Data identifier
    parser.add_argument(
        "--data_id",
        type=str,
        default="aadityasinghal/facial-expression-dataset",
        help="Identifier for the kaggle dataset (default: aadityasinghal/facial-expression-dataset)",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="train/train",
        help="relative path of train dataset (default: train/train)",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="test/test",
        help="relative path of test dataset (default: test/test)",
    )
    # Dataloader
    # batch_size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for dataloaders (default: 128)",
    )
    # num_workers
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for dataloaders (default: 8)",
    )

    # seed set
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed value for randomization (default: 42)",
    )

    return parser.parse_args()


def get_logger(
    local_rank=0,
    name="logger",
    cli_level=logging.INFO,
    log_level=logging.DEBUG,
    cli=True,
    log_dir=None,
):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if cli:
        cli_formatter = logging.Formatter("%(message)s")
        cli_handler = logging.StreamHandler(sys.stdout)
        cli_handler.setLevel(cli_level if local_rank == 0 else logging.CRITICAL)
        cli_handler.setFormatter(cli_formatter)
        logger.addHandler(cli_handler)

    if log_dir:
        log_dir = Path(log_dir)
        if log_dir.is_file():
            raise ValueError(f"Log directory path exists as file: {log_dir}")
        log_dir.mkdir(exist_ok=True, parents=True)

        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(
            log_dir / f"log_rank{local_rank}.txt",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def plot(
    train_acc_list,
    train_loss_list,
    val_acc_list,
    val_loss_list,
    fig_save_dir="./",
):
    fig_save_dir = Path(fig_save_dir)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    axes[0].plot(train_acc_list, label="Train Accuracy")
    axes[0].plot(val_acc_list, label="Validation Accuracy")
    axes[0].legend()
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[1].plot(train_loss_list, label="Train Loss")
    axes[1].plot(val_loss_list, label="Validation Loss")
    axes[1].legend()
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    fig.savefig(fig_save_dir / "accuracy_and_loss.png", dpi=600)
