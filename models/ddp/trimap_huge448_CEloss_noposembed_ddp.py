# model/iter_mask/trimap_huge448_CEloss_noposembed_ddp.py

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from isegm.utils.exp_imports.default import *
from torch.nn import CrossEntropyLoss

from isegm.utils.serialization import get_config_repr

from isegm.data.datasets.aim500 import AIM500TrimapDataset
from isegm.data.datasets.p3m10k import P3M10KTrimapDataset
from isegm.data.datasets.am2k import AM2KTrimapDataset
from isegm.data.datasets.composition import COMPOSITIONTrimapDataset
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from isegm.model.trimap_combineloss import CombinedLoss
from isegm.model.losses import NormalizedFocalLossSoftmax, UnknownRegionDTLoss

MODEL_NAME = 'new_augmentation_trimap_vit_huge448_CE_loss_noposembed_ddp'


def main(cfg):
    #dist.init_process_group(backend='nccl')
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    cfg.device = torch.device(f'cuda:{local_rank}')
    
    model, model_cfg = init_model(cfg)
    
    # === 여기가 중요 ===
    if local_rank == 0:
        logger.info(model)
        logger.info(get_config_repr(model._config))

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    global MODEL_NAME
    MODEL_NAME += f'_{cfg.upsample}'

    train(model, cfg, model_cfg, local_rank)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (cfg.INPUT_SIZE, cfg.INPUT_SIZE)
    model_cfg.num_max_points = 24

    backbone_params = dict(
        img_size=model_cfg.crop_size,
        patch_size=(14,14),
        in_chans=3,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True,
    )

    neck_params = dict(
        in_dim = 1280,
        out_dims = [240, 480, 960, 1920],
    )

    head_params = dict(
        in_channels=[240, 480, 960, 1920],
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=3,
        loss_decode=nn.CrossEntropyLoss(),
        align_corners=False,
        upsample=cfg.upsample,
        channels={'x1': 256, 'x2': 128, 'x4': 64}[cfg.upsample],
    )

    model = NoPosEmbedTrimapPlainVitModel(
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        random_split=cfg.random_split,
    )

    model.backbone.init_weights_from_pretrained(cfg.IMAGENET_PRETRAINED_MODELS.MAE_HUGE)
    model.to(cfg.device)

    return model, model_cfg


def train(model, cfg, model_cfg, local_rank):
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = nn.CrossEntropyLoss()
    loss_cfg.instance_loss_weight = 1.0

    # Dataset
    trainset1 = AM2KTrimapDataset(cfg.AM2K_PATH, 'train', crop_size=crop_size, do_aug=True, epoch_len=-1)
    trainset2 = P3M10KTrimapDataset(cfg.P3M10K_TRAIN_PATH, 'train', crop_size=crop_size, do_aug=True, epoch_len=-1)
    trainset3 = COMPOSITIONTrimapDataset(cfg.COMPOSITION431K_PATH, 'train', crop_size=crop_size, do_aug=True, epoch_len=-1)

    full_train = ConcatDataset([trainset1, trainset2, trainset3])
    train_sampler = DistributedSampler(full_train, shuffle=True)  # DDP용 샘플러

    valset = P3M10KTrimapDataset(cfg.P3M10K_TEST_PATH, 'val', crop_size=crop_size, do_aug=False, epoch_len=-1)

    optimizer_params = {
        'lr': 1e-4,
        'betas': (0.9, 0.999),
        'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR, milestones=[50, 55], gamma=0.1)

    if dist.get_rank() == 0:
        logger.info(f'Trainset1 (AM2K): {len(trainset1)} samples')
        logger.info(f'Trainset2 (P3M10K): {len(trainset2)} samples')
        logger.info(f'Trainset3 (COMPOSITION): {len(trainset3)} samples')

        total_samples = sum([d.get_samples_number() for d in full_train.datasets])
        logger.info(f'Dataset of {total_samples} samples was loaded for training.')

        val_samples = valset.get_samples_number()
        logger.info(f'Dataset of {val_samples} samples was loaded for validation.')

    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        full_train, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        layerwise_decay=cfg.layerwise_decay,
                        lr_scheduler=lr_scheduler,
                        train_sampler=train_sampler,
                        checkpoint_interval=[(0, 10), (50, 1)],
                        image_dump_interval=300,
                        metrics=[PerClassIoU(), MultiClassIoU(), UnknownIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3)

    trainer.run(num_epochs=55, validation=True)