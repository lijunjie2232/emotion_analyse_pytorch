import kagglehub
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model import EmotionNet
from utils import (
    load,
    save,
    val_epoch,
    train_epoch,
    ddp_setup,
    ddp_cleanup,
    parse_args,
    get_logger,
    plot,
)
from tqdm import tqdm
import shutil
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, get_world_size
import os
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")

LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


def main(local_rank, world_size, args):
    # ddp init
    ddp_setup(
        local_rank,
        world_size,
        master_addr="127.0.0.1",
        master_port=29500,
    )

    # logger
    logger = get_logger(local_rank)

    # ## Hyper Parameters and Configs
    # device = "cuda:1"
    device = "cuda"

    amp = args.amp
    lr = args.lr
    step_size = args.step_size
    batch_size = args.batch_size // world_size
    num_workers = args.num_workers
    epochs = args.epochs
    start_epoch = args.start_epoch
    train_patience = args.train_patience
    best_checkpoint = Path(args.best_checkpoint)
    last_checkpoint = Path(args.last_checkpoint)
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path

    # ## cudnn acceleration
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    # ## random seed
    seed = args.seed  # + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ## prepare dataset
    # Download latest version
    data_id = args.data_id
    data_root = kagglehub.dataset_download(data_id)
    logger.info("Path to dataset files: " + data_root)
    data_root = Path(data_root)

    # ## build data transforms
    train_transformer = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),  # 224x224
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    val_transformer = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # ## build dataset
    train_dataset = ImageFolder(
        root=data_root / train_data_path,
        transform=train_transformer,
    )
    val_dataset = ImageFolder(
        root=data_root / val_data_path,
        transform=val_transformer,
    )

    # ## build model
    model = EmotionNet(
        nc=len(train_dataset.classes),
    ).cuda()

    # ## build loss, optimizer and lr_scheduler

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=0.1,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    if last_checkpoint.is_file():
        ckp_epoch = load(
            model=model,
            optimizer=optimizer,
            path=last_checkpoint,
        )
        if start_epoch == -1:
            start_epoch = ckp_epoch
        logger.info(
            f"continue from {last_checkpoint.__str__()}, "
            + f"start at epoch {start_epoch+1}"
        )

    # ## model set ddp
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        # find_unused_parameters=True,
    )

    # ## build dataloader
    train_sampler = DistributedSampler(
        train_dataset,
        rank=get_rank(),
        seed=seed,
        num_replicas=get_world_size(),
    )
    train_sampler.set_epoch(start_epoch + 1)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=train_sampler,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers * 2,
        shuffle=False,
    )

    # ## train model
    loop = (
        tqdm(range(start_epoch + 1, epochs))
        if local_rank == 0
        else range(start_epoch + 1, epochs)
    )
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    best_acc = 0
    patience = train_patience
    for epoch in loop:
        train_sampler.set_epoch(epoch)
        if local_rank == 0:
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        # torch.distributed.barrier()
        train_acc, train_loss = train_epoch(
            ddp_model,
            train_dataloader,
            optimizer,
            scheduler,
            criterion,
            scaler,
            device=device,
            progress=local_rank == 0,
            fp16=amp,
        )
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        if local_rank == 0:
            loop.set_postfix(
                train_acc=train_acc,
                train_loss=train_loss,
            )
            val_acc, val_loss = val_epoch(
                ddp_model,
                val_dataloader,
                criterion,
                device=device,
            )
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            loop.set_postfix(
                train_acc=train_acc,
                train_loss=train_loss,
                val_acc=val_acc,
                val_loss=val_loss,
            )

            save(ddp_model.module, optimizer, epoch, last_checkpoint)

            plot(
                train_acc_list,
                train_loss_list,
                val_acc_list,
                val_loss_list,
                fig_save_dir="./",
            )
            if val_acc > best_acc:
                shutil.copyfile(last_checkpoint, best_checkpoint)
                best_acc = val_acc
                patience = train_patience
            else:
                patience -= 1
            if patience == 0:
                logger.info("no patience, early stop...")
                break
    ddp_cleanup()
    logger.info("done.")


if __name__ == "__main__":

    args = parse_args()
    main(LOCAL_RANK, WORLD_SIZE, args)
