import os
import random
import logging
from copy import deepcopy
from collections import defaultdict

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from isegm.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.vis import draw_probmap, draw_points
from isegm.utils.misc import save_checkpoint
from isegm.utils.serialization import get_config_repr
from isegm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict
from .optimizer import get_optimizer, get_optimizer_with_layerwise_decay


class ISTrainer(object):
    def __init__(self, model, cfg, model_cfg, loss_cfg,
                 trainset, valset,
                 optimizer='adam',
                 optimizer_params=None,
                 layerwise_decay=False,
                 image_dump_interval=200,
                 checkpoint_interval=10,
                 tb_dump_period=25,
                 max_interactive_points=0,
                 lr_scheduler=None,
                 metrics=None,
                 additional_val_metrics=None,
                 net_inputs=('images', 'seg_mask'),
                 max_num_next_clicks=0,
                 click_models=None,
                 prev_mask_drop_prob=0.0,
                 ):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.net_inputs = net_inputs
        self.max_num_next_clicks = max_num_next_clicks

        self.click_models = click_models
        self.prev_mask_drop_prob = prev_mask_drop_prob

        if cfg.distributed:
            cfg.batch_size //= cfg.ngpus
            cfg.val_batch_size //= cfg.ngpus

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ''
        self.sw = None

        self.trainset = trainset
        self.valset = valset

        #logger.info(f'Dataset of {trainset.get_samples_number()} samples was loaded for training.')
        
        if hasattr(trainset, 'get_samples_number'):
            total_train_samples = trainset.get_samples_number()
        else:
            total_train_samples = sum(d.get_samples_number() for d in trainset.datasets)

        logger.info(f'Dataset of {total_train_samples} samples was loaded for training.')        
        logger.info(f'Dataset of {valset.get_samples_number()} samples was loaded for validation.')

        self.train_data = DataLoader(
            trainset, cfg.batch_size,
            sampler=get_sampler(trainset, shuffle=True, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        self.val_data = DataLoader(
            valset, cfg.val_batch_size,
            sampler=get_sampler(valset, shuffle=False, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        if layerwise_decay:
            self.optim = get_optimizer_with_layerwise_decay(model, optimizer, optimizer_params)
        else:
            self.optim = get_optimizer(model, optimizer, optimizer_params)
        model = self._load_weights(model)

        if cfg.multi_gpu:
            model = get_dp_wrapper(cfg.distributed)(model, device_ids=cfg.gpu_ids,
                                                    output_device=cfg.gpu_ids[0])

        if self.is_master:
            logger.info(model)
            logger.info(get_config_repr(model._config))

        self.device = cfg.device
        self.net = model.to(self.device)
        self.lr = optimizer_params['lr']

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        if self.click_models is not None:
            for click_model in self.click_models:
                for param in click_model.parameters():
                    param.requires_grad = False
                click_model.to(self.device)
                click_model.eval()

    def run(self, num_epochs, start_epoch=None, validation=True):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch

        logger.info(f'Starting Epoch: {start_epoch}')
        logger.info(f'Total Epochs: {num_epochs}')
        for epoch in range(start_epoch, num_epochs):
            self.training(epoch)
            if validation:
                self.validation(epoch)

    def training(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        if self.cfg.distributed:
            self.train_data.sampler.set_epoch(epoch)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100)\
            if self.is_master else self.train_data

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        self.net.train()
        train_loss = 0.0
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.train_data) + i

            loss, losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses_logging['overall'] = loss
            reduce_loss_dict(losses_logging)

            train_loss += losses_logging['overall'].item()

            if self.is_master:
                for loss_name, loss_value in losses_logging.items():
                    self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}',
                                       value=loss_value.item(),
                                       global_step=global_step)

                for k, v in self.loss_cfg.items():
                    if '_loss' in k and hasattr(v, 'log_states') and self.loss_cfg.get(k + '_weight', 0.0) > 0:
                        v.log_states(self.sw, f'{log_prefix}Losses/{k}', global_step)

                # if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                if self.image_dump_interval > 0 and (global_step % self.image_dump_interval == 0 or i == len(self.train_data) - 1):
                    self.save_visualization(splitted_batch_data, outputs, global_step, prefix='train')

                self.sw.add_scalar(tag=f'{log_prefix}States/learning_rate',
                                   value=self.lr if not hasattr(self, 'lr_scheduler') else self.lr_scheduler.get_lr()[-1],
                                   global_step=global_step)

                tbar.set_description(f'Epoch {epoch}, training loss {train_loss/(i+1):.4f}')
                for metric in self.train_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for metric in self.train_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}',
                                   value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

            save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                            epoch=None, multi_gpu=self.cfg.multi_gpu)

            if isinstance(self.checkpoint_interval, (list, tuple)):
                checkpoint_interval = [x for x in self.checkpoint_interval if x[0] <= epoch][-1][1]
            else:
                checkpoint_interval = self.checkpoint_interval

            if epoch % checkpoint_interval == 0:
                save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                                epoch=epoch, multi_gpu=self.cfg.multi_gpu)

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

    def validation(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100) if self.is_master else self.val_data

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.eval()
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.val_data) + i
            loss, batch_losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data, validation=True)

            batch_losses_logging['overall'] = loss
            reduce_loss_dict(batch_losses_logging)
            for loss_name, loss_value in batch_losses_logging.items():
                losses_logging[loss_name].append(loss_value.item())

            val_loss += batch_losses_logging['overall'].item()

            if self.is_master:
                tbar.set_description(f'Epoch {epoch}, validation loss: {val_loss/(i + 1):.4f}')
                for metric in self.val_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}', value=np.array(loss_values).mean(),
                                   global_step=epoch, disable_avg=True)

            for metric in self.val_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}', value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

    def batch_forward(self, batch_data, validation=False):
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = dict()

        with torch.set_grad_enabled(not validation):
            # 모든 입력 텐서를 GPU로
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            image = batch_data['images']              # [B, 3, H, W]
            seg_mask = batch_data['seg_mask']         # [B, 1, H, W]
            gt_trimap = batch_data['instances']       # [B, H, W]

            # prev_output, points 관련 불필요한 코드 제거함

            # 모델 forward
            output = self.net(image, seg_mask)
            
            target_size = gt_trimap.shape[-2:]  # (H, W)
            if output['instances'].shape[-2:] != target_size:
                output['instances'] = torch.nn.functional.interpolate(
                    output['instances'],
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )

            if output['instances_aux'] is not None and output['instances_aux'].shape[-2:] != target_size:
                output['instances_aux'] = torch.nn.functional.interpolate(
                    output['instances_aux'],
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
                
            loss = 0.0

            # loss 계산
            loss = self.add_loss('instance_loss', loss, losses_logging, validation,
                                lambda: (output['instances'], gt_trimap))
            loss = self.add_loss('instance_aux_loss', loss, losses_logging, validation,
                                lambda: (output['instances_aux'], gt_trimap))

            if self.is_master:
                with torch.no_grad():
                    for m in metrics:
                        m.update(*(output.get(x) for x in m.pred_outputs),
                                *(batch_data[x] for x in m.gt_outputs))

        return loss, losses_logging, batch_data, output

    def add_loss(self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = torch.mean(loss)
            losses_logging[loss_name] = loss
            loss = loss_weight * loss
            total_loss = total_loss + loss

        return total_loss

    # def save_visualization(self, splitted_batch_data, outputs, global_step, prefix):
    #     #kjk
    #     output_images_path = self.cfg.VIS_PATH / prefix
    #     if self.task_prefix:
    #         output_images_path /= self.task_prefix

    #     output_images_path.mkdir(parents=True, exist_ok=True)
    #     image_name_prefix = f'{global_step:06d}'

    #     def _save_image(suffix, image):
    #         cv2.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.png'),
    #                     image, [cv2.IMWRITE_PNG_COMPRESSION, 3])  # PNG 저장

    #     # 입력데이터
    #     images = splitted_batch_data['images']                   # [B, 3, H, W]
    #     gt_trimap = splitted_batch_data['instances']             # [B, H, W]
    #     pred_logits = outputs['instances']                       # [B, 3, H, W]
    #     pred_trimap = torch.argmax(pred_logits, dim=1)           # [B, H, W]

    #     # 첫번째 배치만 저장
    #     image = images[0].cpu().numpy().transpose(1, 2, 0) * 255
    #     image = image.astype(np.uint8)
    #     gt_mask = gt_trimap[0].cpu().numpy()        # [H, W] with class indices (0,1,2)
    #     pred_mask = pred_trimap[0].cpu().numpy()    # [H, W]

    #     # class index -> trimap 값 매핑
    #     def map_trimap_values(mask):
    #         mapped = np.zeros_like(mask, dtype=np.uint8)
    #         mapped[mask == 0] = 0     # background
    #         mapped[mask == 1] = 128   # unknown
    #         mapped[mask == 2] = 255   # foreground
    #         return mapped

    #     gt_mask = map_trimap_values(gt_mask)
    #     pred_mask = map_trimap_values(pred_mask)

    #     _save_image('gt_trimap', gt_mask)
    #     _save_image('pred_trimap', pred_mask)

    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix):
        output_images_path = self.cfg.VIS_PATH / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix

        output_images_path.mkdir(parents=True, exist_ok=True)

        images = splitted_batch_data['images']                   # [B, 3, H, W]
        seg_masks = splitted_batch_data['seg_mask']              # [B, 1, H, W]
        gt_trimap = splitted_batch_data['instances']             # [B, H, W]
        pred_logits = outputs['instances']                       # [B, 3, H, W]
        pred_trimap = torch.argmax(pred_logits, dim=1)           # [B, H, W]

        batch_size = images.shape[0]
        num_to_save = max(1, batch_size // 4)                    # 전체 배치 중 1/4 저장

        def map_trimap_values(mask):
            mapped = np.zeros_like(mask, dtype=np.uint8)
            mapped[mask == 0] = 0     # background
            mapped[mask == 1] = 128   # unknown
            mapped[mask == 2] = 255   # foreground
            return mapped

        def map_seg_mask(mask):
            mapped = np.zeros_like(mask, dtype=np.uint8)
            mapped[mask != 0] = 255
            return mapped

        for i in range(num_to_save):
            image_name_prefix = f'{global_step:06d}_{i:02d}'

            # Original image
            image = images[i].cpu().numpy().transpose(1, 2, 0) * 255
            image = image.astype(np.uint8)

            # Segmentation mask (aux)
            seg_mask = map_seg_mask(seg_masks[i].cpu().numpy().squeeze())

            # GT trimap (0/128/255) and predicted trimap
            gt_mask = map_trimap_values(gt_trimap[i].cpu().numpy())
            pred_mask = map_trimap_values(pred_trimap[i].cpu().numpy())

            def _save_image(suffix, image):
                cv2.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.png'),
                            image, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            _save_image('image', image)
            _save_image('seg_mask', seg_mask)
            _save_image('gt_trimap', gt_mask)
            _save_image('pred_trimap', pred_mask)


    
    def _load_weights(self, net):
        if self.cfg.weights is not None:
            if os.path.isfile(self.cfg.weights):
                load_weights(net, self.cfg.weights)
                self.cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{self.cfg.weights}'")
        elif self.cfg.resume_exp is not None:
            checkpoints = list(self.cfg.CHECKPOINTS_PATH.glob(f'{self.cfg.resume_prefix}*.pth'))
            assert len(checkpoints) == 1

            checkpoint_path = checkpoints[0]
            logger.info(f'Load checkpoint from path: {checkpoint_path}')
            load_weights(net, str(checkpoint_path))
        return net

    @property
    def is_master(self):
        return self.cfg.local_rank == 0


def get_next_points(pred, gt, points, click_indx, pred_thresh=0.49):
    assert click_indx > 0
    pred = pred.cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))]
            if is_positive:
                points[bindx, num_points - click_indx, 0] = float(coords[0])
                points[bindx, num_points - click_indx, 1] = float(coords[1])
                points[bindx, num_points - click_indx, 2] = float(click_indx)
            else:
                points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                points[bindx, 2 * num_points - click_indx, 2] = float(click_indx)

    return points


def load_weights(model, path_to_weights):
    current_state_dict = model.state_dict()
    new_state_dict = torch.load(path_to_weights, map_location='cpu')['state_dict']
    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict)
