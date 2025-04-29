#model/iter_mask/trimap_huge448.py

from isegm.utils.exp_imports.default import *
from torch.nn import CrossEntropyLoss

from isegm.data.datasets.aim500 import AIM500TrimapDataset
from isegm.data.datasets.p3m10k import P3M10KTrimapDataset
from isegm.data.datasets.am2k import AM2KTrimapDataset
from isegm.data.datasets.composition import COMPOSITIONTrimapDataset
from torch.utils.data import ConcatDataset

from isegm.model.trimap_combineloss import CombinedLoss
from isegm.model.losses import NormalizedFocalLossSoftmax, UnknownRegionDTLoss

MODEL_NAME = 'trimap_vit_huge448_CE_loss'


def main(cfg):
    model, model_cfg = init_model(cfg)
    global MODEL_NAME
    MODEL_NAME += f'_{cfg.upsample}' # upsample (default : x4)를 모델 명 뒤에 붙여줌
    train(model, cfg, model_cfg)


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
        # num_classes=1,
        num_classes=3,
        # loss_decode=CrossEntropyLoss(),
        loss_decode=nn.CrossEntropyLoss(),
        align_corners=False,
        upsample=cfg.upsample,
        channels={'x1': 256, 'x2': 128, 'x4': 64}[cfg.upsample],
    )

    model = TrimapPlainVitModel(
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


def train(model, cfg, model_cfg):
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = nn.CrossEntropyLoss()
    loss_cfg.instance_loss_weight = 1.0
    
    train_augmentator = Compose([
        LongestMaxSize(max_size=cfg.INPUT_SIZE),
        HorizontalFlip(),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
        PadIfNeeded(min_height=cfg.INPUT_SIZE, min_width=cfg.INPUT_SIZE, border_mode=0),  # zero-padding
    ], p=1.0)

    val_augmentator = Compose([
        LongestMaxSize(max_size=cfg.INPUT_SIZE),        
        PadIfNeeded(min_height=cfg.INPUT_SIZE, min_width=cfg.INPUT_SIZE, border_mode=0)
    ], p=1.0)

    trainset1 = AM2KTrimapDataset(
        dataset_path=cfg.AM2K_PATH,
        split='train',
        augmentator=train_augmentator,
        epoch_len=-1
    )
    trainset2 = P3M10KTrimapDataset(
        dataset_path=cfg.P3M10K_TRAIN_PATH,
        split='train',
        augmentator=train_augmentator,
        epoch_len=-1
    )
    trainset3 = COMPOSITIONTrimapDataset(
        dataset_path=cfg.COMPOSITION431K_PATH,
        split='train',
        augmentator=train_augmentator,
        epoch_len=-1
    )
    
    trainset = ConcatDataset([trainset1, trainset2, trainset3])

    valset = P3M10KTrimapDataset(
        dataset_path=cfg.P3M10K_TEST_PATH,
        split='val',
        augmentator=val_augmentator,
        epoch_len=-1
    )

    optimizer_params = {
        'lr': 1e-4, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[50, 55], gamma=0.1)
    
    logger.info(f'Trainset1 (AIM500): {len(trainset1)} samples')
    logger.info(f'Trainset2 (P3M10K): {len(trainset2)} samples')
    logger.info(f'Trainset3 (COMPOSITION): {len(trainset3)} samples')
    
    total_samples = sum([d.get_samples_number() for d in trainset.datasets])
    logger.info(f'Dataset of {total_samples} samples was loaded for training.')

    val_samples = valset.get_samples_number()
    logger.info(f'Dataset of {val_samples} samples was loaded for validation.')
    
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        layerwise_decay=cfg.layerwise_decay,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 10), (50, 1)],
                        image_dump_interval=300, # interval to save png
                        metrics=[PerClassIoU(), MultiClassIoU(), UnknownIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3)
    trainer.run(num_epochs=55, validation=False)
    