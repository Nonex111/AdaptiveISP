import pickle as pickle
import os
import shutil
import datetime
import cv2
import numpy as np
import yaml
from tqdm import tqdm
import importlib
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, distributed
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("yolov3")
from yolov3.utils.dataloaders import InfiniteDataLoader, LoadImagesAndLabels
from yolov3.utils.loss import ComputeLoss, ComputeLossBatch
from yolov3.utils.general import LOGGER, colorstr, intersect_dicts, check_dataset, TQDM_BAR_FORMAT, check_img_size, \
    labels_to_class_weights
from yolov3.utils.torch_utils import torch_distributed_zero_first
from yolov3.utils.dataloaders import seed_worker
from yolov3.utils.downloads import attempt_download
from yolov3.models.yolo import Model
from yolov3.utils.callbacks import Callbacks
from yolov3.utils.autoanchor import check_anchors
from yolov3.utils.metrics import fitness
from yolov3.utils.torch_utils import EarlyStopping
from yolov3.models.experimental import attempt_load

from replay_memory import ReplayMemory, create_input_tensor
from util import make_image_grid, Tee, merge_dict, Dict, save_img
from util import STATE_DROPOUT_BEGIN, STATE_REWARD_DIM, STATE_STEP_DIM, STATE_STOPPED_DIM
from agent import Agent
from value import Value
from thermal_branch import ThermalBaselineISP
# from config import cfg
from dataloader import (
    get_noise,
    get_initial_states,
    create_dataloader_real_hr,
    create_dataloader,
    create_dataloader_real,
)
from isp.fixed_pipeline import FixedISPPipeline


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders
WORLD_SIZE = 1


import matplotlib.pyplot as plt
def show(x, title="a", format="HWC", is_last=True):
    if format == 'CHW':
        x = np.transpose(x, (1, 2, 0))
    plt.figure()
    plt.cla()
    plt.title(title)
    plt.imshow(x)
    if is_last:
        plt.show()


class DynamicISP:
    def __init__(self, args, task="train_val"):
        train = False
        val = False
        if task == "train":
            train = True
        elif task == "train_val":
            train = True
            val = True
        if train:
            self.base_dir = os.path.join('experiments', args.save_path)
            os.makedirs(self.base_dir, exist_ok=True)
            self.log_dir = os.path.join(self.base_dir, "logs")
            os.makedirs(self.log_dir, exist_ok=True)
            self.ckpt_dir = os.path.join(self.base_dir, "ckpt")
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.tee = Tee(os.path.join(self.log_dir, 'log.txt'))
            self.writer = SummaryWriter(self.log_dir)
            self.image_dir = os.path.join(self.base_dir, "images")
            os.makedirs(self.image_dir, exist_ok=True)

            shutil.copy(f"config.py", os.path.join(self.base_dir, f"config.py"))
            print("Training begin....")
            print("------- Baseline + Truncated ---------")
        
        try:
            cfg = importlib.import_module(f'{args.cfg}').cfg
        except Exception as e:
            print(e)
            print(f"don't support {args.cfg}!")

        self.device = torch.device('cuda')
        cfg.filter_runtime_penalty = args.runtime_penalty
        cfg.filter_runtime_penalty_lambda = args.runtime_penalty_lambda
        if getattr(args, "masking", None) is not None:
            cfg.masking = args.masking
        if getattr(args, "ifcnn_weights", None) is not None:
            cfg.ifcnn_weights = args.ifcnn_weights
        if getattr(args, "ifcnn_trainable", None) is not None:
            cfg.ifcnn_trainable = args.ifcnn_trainable
        if getattr(args, "ifcnn_fuse_scheme", None) is not None:
            cfg.ifcnn_fuse_scheme = int(args.ifcnn_fuse_scheme)
        if getattr(args, "force_filter_name", None) is not None:
            cfg.force_filter_name = args.force_filter_name
        if getattr(args, "force_filter_step", None) is not None:
            cfg.force_filter_step = int(args.force_filter_step) if int(args.force_filter_step) >= 0 else None

        # Optional constraints for NLM strength (kept off by default to preserve original behavior).
        try:
            nlm_limit = float(getattr(args, "nlm_limit", -1.0))
        except Exception:
            nlm_limit = -1.0
        cfg.nlm_limit = float(nlm_limit) if nlm_limit >= 0 else None
        try:
            nlm_init = float(getattr(args, "nlm_init", -1.0))
        except Exception:
            nlm_init = -1.0
        cfg.nlm_init = nlm_init if nlm_init > 0 else None

        # --- ISP mode tweaks (reuse RL code path) ---
        isp_mode = getattr(args, "isp_mode", "rl")
        if isp_mode == "fixed_postprocess":
            # 1) Fix WB/CCM upstream (camera metadata), so remove them from the RL action space.
            remove_filter_names = {"CCMFilter", "ImprovedWhiteBalanceFilter"}
            orig_filters = list(getattr(cfg, "filters", []))
            orig_runtime = list(getattr(cfg, "filters_runtime", []))
            new_filters = []
            new_runtime = []
            runtime_aligned = len(orig_runtime) == len(orig_filters)
            for i, f in enumerate(orig_filters):
                if getattr(f, "__name__", "") in remove_filter_names:
                    continue
                new_filters.append(f)
                if runtime_aligned:
                    new_runtime.append(orig_runtime[i])
            cfg.filters = new_filters
            if hasattr(cfg, "filters_runtime"):
                cfg.filters_runtime = new_runtime if runtime_aligned else [0.0 for _ in new_filters]

            # 2) Optionally force denoise first (learnable params), then run an RL-chosen post pipeline.
            # If disabled, fixed_postprocess becomes "RL postprocess" with a reduced action space (no WB/CCM),
            # and --steps is interpreted as the total number of RL steps (same as rl mode).
            force_nlm_first = bool(getattr(args, "force_nlm_first_in_fixed_postprocess_mode", True))

            post_steps = int(getattr(args, "steps", getattr(cfg, "test_steps", 5)))
            cfg.test_steps = post_steps + 1 if force_nlm_first else post_steps
            if getattr(cfg, "maximum_trajectory_length", 0) < cfg.test_steps:
                cfg.maximum_trajectory_length = cfg.test_steps

            # Optionally force Gamma as the last step (typical ISP order: ... -> gamma).
            force_schedule = {0: "NLM"} if force_nlm_first else {}
            if getattr(args, "fixed_postprocess_force_gamma", True):
                force_schedule[int(cfg.test_steps) - 1] = "G"
                gamma_init = float(getattr(args, "gamma_init", 2.2))
                if gamma_init > 0:
                    cfg.gamma_init = gamma_init
            if force_schedule:
                cfg.force_filter_schedule = force_schedule

            # If we force NLM at step 0, disallow selecting it later to keep the "postprocess-only" steps clean.
            if force_nlm_first:
                cfg.disallow_filter_short_names_after_step0 = ["NLM"]

            # Update dependent dims after filter list change.
            cfg.num_state_dim = 3 + len(cfg.filters)
            cfg.z_dim = 3 + len(cfg.filters) * int(getattr(cfg, "z_dim_per_filter", 16))

        # --- VIF: infrared branch mode ---
        self.ir_branch_mode = getattr(args, "ir_branch_mode", "baseline")
        cfg.ir_branch_mode = self.ir_branch_mode
        if self.ir_branch_mode == "dual":
            # Dual-branch decision/state: action space includes IR filters, and we fuse (IFCNN) after every step.
            from isp.filters import IRAGCFilter, IRGaussianDenoiseFilter, IRLogToneMapFilter

            cfg.fuse_after_each_step = True
            cfg.include_ir_in_agent = True
            cfg.include_ir_in_value = True

            orig_filters = list(getattr(cfg, "filters", []))
            orig_runtime = list(getattr(cfg, "filters_runtime", []))
            runtime_map = {}
            for i, f in enumerate(orig_filters):
                if i < len(orig_runtime):
                    runtime_map[getattr(f, "__name__", str(f))] = orig_runtime[i]

            drop_names = {"IFCNNFusionFilter", "MaxFusionFilter", "MeanFusionFilter"}
            base_filters = [f for f in orig_filters if getattr(f, "__name__", "") not in drop_names]
            cfg.filters = [IRGaussianDenoiseFilter, IRLogToneMapFilter, IRAGCFilter] + base_filters
            cfg.filters_runtime = [
                runtime_map.get(getattr(f, "__name__", str(f)), 0.5) for f in cfg.filters
            ]

            cfg.num_state_dim = 3 + len(cfg.filters)
            cfg.z_dim = 3 + len(cfg.filters) * int(getattr(cfg, "z_dim_per_filter", 16))
        else:
            cfg.fuse_after_each_step = False
            cfg.include_ir_in_agent = False
            cfg.include_ir_in_value = False

        # Hyperparameters
        hyp = args.hyp
        if isinstance(hyp, str):
            with open(hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
        LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
        args.hyp = hyp.copy()  # for saving hyps to checkpoints
        data_dict = check_dataset(args.data_cfg)
        nc = int(data_dict['nc'])  # number of classes
        resume = False

        # Load Pretrained YOLO
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(args.weights)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        yolo_model = Model(args.yolo_cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(
            self.device)  # create
        exclude = ['anchor'] if (args.yolo_cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, yolo_model.state_dict(), exclude=exclude)  # intersect
        yolo_model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(yolo_model.state_dict())} items from {weights}')  # report

        # Data Loader  TODO train
        train_path, val_path = data_dict['train'], data_dict['val']
        if task == "test":
            val_path = data_dict['test']
        # Image size
        gs = max(int(yolo_model.stride.max()), 32)  # grid size (max stride)
        args.imgsz = check_img_size(args.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
        # if train:
        self.train_loader = ReplayMemory(cfg, train, train_path, args.imgsz, args.batch_size, gs,
                                            single_cls=False, hyp=hyp, augment=False, cache=False, pad=0.0,
                                            rect=False, image_weights=False, prefix=colorstr('train: '), limit=-1,
                                            add_noise=args.add_noise, data_name=args.data_name, brightness_range=args.bri_range,
                                            noise_level=args.noise_level, use_linear=args.use_linear,
                                            vi_dir_name=getattr(args, "vi_dir_name", "vi"),
                                            ir_dir_name=getattr(args, "ir_dir_name", "ir"),
                                            ir_root=getattr(args, "ir_root", None),
                                            ir_use_y16=getattr(args, "ir_use_y16", True),
                                            ir_width=getattr(args, "ir_width", None),
                                            ir_height=getattr(args, "ir_height", None),
                                            apply_meta_wb_ccm=getattr(args, "apply_meta_wb_ccm", False))
        if val:
            self.val_loader = ReplayMemory(cfg, val, val_path, args.imgsz, args.batch_size, gs,
                                           single_cls=False, hyp=hyp, augment=False, cache=False, pad=0.0,
                                           rect=False, image_weights=False, prefix=colorstr('val: '), limit=-1,
                                           add_noise=args.add_noise, data_name=args.data_name, brightness_range=args.bri_range,
                                           noise_level=args.noise_level, use_linear=args.use_linear,
                                           vi_dir_name=getattr(args, "vi_dir_name", "vi"),
                                           ir_dir_name=getattr(args, "ir_dir_name", "ir"),
                                           ir_root=getattr(args, "ir_root", None),
                                           ir_use_y16=getattr(args, "ir_use_y16", True),
                                           ir_width=getattr(args, "ir_width", None),
                                           ir_height=getattr(args, "ir_height", None),
                                           apply_meta_wb_ccm=getattr(args, "apply_meta_wb_ccm", False))
            self.val_loader = self.val_loader.get_feed_dict_and_states(8)
        # Model attributes
        # check_anchors(dataset, model=yolo_model, thr=hyp['anchor_t'], imgsz=args.imgsz)  # run AutoAnchor
        nl = yolo_model.model[-1].nl  # number of detection layers (to scale hyps)  3
        hyp['box'] *= 3 / nl  # scale to layers
        hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        hyp['obj'] *= (args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        hyp['label_smoothing'] = 0.0
        yolo_model.nc = nc  # attach number of classes to model
        yolo_model.hyp = hyp  # attach hyperparameters to model
        yolo_model.class_weights = labels_to_class_weights(self.train_loader.dataset.labels, nc).to(self.device) * nc  # attach class weights
        yolo_model.names = data_dict['names']  # class names
        yolo_model = yolo_model.to(self.device)
        self.yolo_model = yolo_model
        self.data_dict = data_dict

        # Ensure optional cfg keys exist to avoid Dict.__getattr__ KeyError.
        if "force_filter_schedule" not in cfg:
            cfg.force_filter_schedule = None
        if "disallow_filter_short_names_after_step0" not in cfg:
            cfg.disallow_filter_short_names_after_step0 = None

        if "num_state_dim" not in cfg:
            cfg.num_state_dim = 3 + len(cfg.filters)
        if "z_dim" not in cfg:
            cfg.z_dim = 3 + len(cfg.filters) * int(getattr(cfg, "z_dim_per_filter", 16))

        base_agent_ch = 3 + (1 if bool(getattr(cfg, "include_ir_in_agent", False)) else 0)
        agent_in_ch = base_agent_ch + (cfg.num_state_dim if cfg.img_include_states else 0)
        base_value_ch = 3 + (1 if bool(getattr(cfg, "include_ir_in_value", False)) else 0)
        value_in_ch = base_value_ch + cfg.num_state_dim + 3

        self.agent = Agent(cfg, shape=(agent_in_ch, 64, 64), meta_ccm=cfg.meta_ccm).to(self.device)
        self.value = Value(cfg, shape=(value_in_ch, 64, 64)).to(self.device)
        self.thermal_isp = None
        if args.data_name in ("vif",) and self.ir_branch_mode == "baseline":
            self.thermal_isp = ThermalBaselineISP(cfg, device=str(self.device)).to(self.device)
        self.args = args
        cfg.max_iter_step = int(self.args.epochs * 1000 // args.batch_size)  # 1000 train images
        if cfg.show_img_num > args.batch_size:
            cfg.show_img_num = args.batch_size

        self.gs = gs
        self.hyp = hyp
        self.val_path = val_path
        self.filter_name = [x.get_short_name() for x in self.agent.filters]

        print("----------------- args ------------------")
        for k, v in vars(args).items():
            print(k, ":", v)
        print("---------------- config ------------------")
        for k, v in cfg.items():
            print(k, ":", v)
        self.cfg = cfg

        self.max_bri = 0.9 # 0.8

    @staticmethod
    def compute_loss_batch(func, preds, targets, device):
        # for x in preds:
        #     print(x.shape)
        # print(targets_tensor.shape)
        batch = preds[0].shape[0]
        lclss = torch.zeros((batch, 1), device=device)  # class loss
        lboxs = torch.zeros((batch, 1), device=device)  # box loss
        lobjs = torch.zeros((batch, 1), device=device)  # object loss
        for b in range(batch):
            pred_one = []
            for i in range(len(preds)):
                pred_one.append(preds[i][b].unsqueeze(0).to(device))
            target_one = targets[b]
            target_one[:, 0] = 0
            # for x in pred_one:
            #     print(x.shape)
            # print(target_one.shape)
            lbox, lobj, lcls = func(pred_one, target_one.to(device))
            lboxs[b] = lbox
            lobjs[b] = lobj
            lclss[b] = lcls
        return lboxs + lobjs + lclss, torch.cat((lboxs, lobjs, lclss)).detach()

    def train(self):
        if self.args.resume is not None:
            print(f"Resume from {self.args.resume}")
            ckpt = torch.load(self.args.resume)
            self.agent.load_state_dict(ckpt['agent_model'])
            self.value.load_state_dict(ckpt['value_model'])
            if self.thermal_isp is not None and 'thermal_model' in ckpt:
                try:
                    self.thermal_isp.load_state_dict(ckpt['thermal_model'])
                except Exception:
                    pass

        agent_params = list(self.agent.parameters())
        if self.thermal_isp is not None:
            agent_params += list(self.thermal_isp.parameters())
        agent_optimizer = torch.optim.Adam(agent_params,
                                           lr=self.args.lr)  # , betas=(0.5, 0.9)
        value_optimizer = torch.optim.Adam(self.value.parameters(),
                                           lr=self.args.lr * float(self.cfg.value_lr_mul))  # , betas=(0.5, 0.9)
        lr_decay = 0.1
        # base_lr = self.args.lr
        # agent_lr_mul = 0.3
        segments = 3
        max_iter_step = self.cfg.max_iter_step
        agent_lr = lambda iter: lr_decay ** (1.0 * iter * segments / max_iter_step)
        value_lr = lambda iter: lr_decay ** (1.0 * iter * segments / max_iter_step)
        agent_scheduler = torch.optim.lr_scheduler.LambdaLR(agent_optimizer, lr_lambda=agent_lr)
        value_scheduler = torch.optim.lr_scheduler.LambdaLR(value_optimizer, lr_lambda=value_lr)
        # agent_scheduler = torch.optim.lr_scheduler.ExponentialLR(agent_optimizer, gamma=lr_decay, last_epoch=-1)
        # value_scheduler = torch.optim.lr_scheduler.ExponentialLR(value_optimizer, gamma=lr_decay, last_epoch=-1)
        print("'init learning rate, agent:", agent_scheduler.get_lr()[0], " value:", value_scheduler.get_lr()[0])

        compute_loss = ComputeLoss(self.yolo_model)
        compute_loss_batch = ComputeLossBatch(self.yolo_model, reduction="mean")  # sum
        callbacks = Callbacks()
        callbacks.run('on_train_start')

        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.args.workers} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.args.save_path)}\n"
                    f'Starting training for {0} epochs...')
        mloss_agent, mloss_value, mloss_detect = 0.0, 0.0, 0.0

        for iter in range(self.cfg.max_iter_step+1):
            self.agent.train()
            self.value.train()
            self.yolo_model.train()
            # fixed all layers
            for k, v in self.yolo_model.named_parameters():
                v.requires_grad = False
            for m in self.yolo_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            progress = float(iter) / self.cfg.max_iter_step
            feed_dict = self.train_loader.get_feed_dict_and_states(self.args.batch_size)
            # "im", "label", "path", "shape", "state", "z": numpy
            imgs, targets, paths, shapes, states = create_input_tensor(
                (feed_dict['im'], feed_dict['label'], feed_dict['path'], feed_dict['shape'], feed_dict['state']))
            extra = None
            ir_imgs = None
            ir_before = None
            ir_after = None
            if isinstance(imgs, (tuple, list)) and len(imgs) == 2:
                imgs, ir_imgs = imgs
            z = torch.from_numpy(feed_dict['z']).to(self.device)

            # callbacks.run('on_train_batch_start')
            imgs = imgs.to(self.device, non_blocking=True).float()  # input 0.0-1.0
            states = states.to(self.device)
            if ir_imgs is not None:
                ir_before = ir_imgs.to(self.device, non_blocking=True).float()
                if self.ir_branch_mode == "baseline" and self.thermal_isp is not None:
                    ir_after, _ = self.thermal_isp(ir_before)
                    extra = {"ir": ir_after}
                else:
                    extra = {"ir": ir_before}

            # Forward
            agent_out, agent_debug_out, agent_debugger = self.agent((imgs, z, states, extra), progress)
            retouch, new_states, surrogate, penalty = agent_out
            stopped = new_states[:, STATE_STOPPED_DIM:STATE_STOPPED_DIM + 1]
            if self.ir_branch_mode == "dual" and isinstance(agent_debug_out, dict) and agent_debug_out.get("ir", None) is not None:
                ir_after = agent_debug_out["ir"]
            if ir_before is not None and ir_after is None:
                ir_after = ir_before

            pred_input = self.yolo_model(imgs)
            # detect_input_loss, detect_input_loss_items = compute_loss(pred_input, targets.to(self.device))  # loss scaled by batch_size
            detect_input_loss_raw, _ = self.compute_loss_batch(compute_loss_batch, pred_input, feed_dict['label'], self.device)
            detect_input_loss = torch.clip(detect_input_loss_raw * self.cfg.detect_loss_weight, 0, 1.0)

            pred_retouch = self.yolo_model(retouch)
            # detect_retouch_loss, detect_retouch_loss_items = compute_loss(pred_retouch, targets.to(self.device))  # loss scaled by batch_size
            _, detect_retouch_loss_items = compute_loss(pred_retouch, targets.to(self.device))  # loss scaled by batch_size
            detect_retouch_loss_raw, _ = self.compute_loss_batch(compute_loss_batch, pred_retouch, feed_dict['label'], self.device)  # loss scaled by batch_size
            detect_retouch_loss = torch.clip(detect_retouch_loss_raw * self.cfg.detect_loss_weight, 0, 1.0)

            input_mean = torch.mean(imgs.detach(), dim=(1, 2, 3)).unsqueeze(-1)  # [B, 1]
            retouch_mean = torch.mean(retouch.detach(), dim=(1, 2, 3)).unsqueeze(-1)  # [B, 1]
            retouch_nonfinite = ~torch.isfinite(retouch_mean)
            dark_threshold = retouch_mean.new_tensor(float(self.args.retouch_dark_abs))
            if self.args.relative_brightness:
                dark_threshold = torch.minimum(dark_threshold, input_mean * float(self.args.retouch_dark_ratio))
            retouch_too_dark = retouch_mean < dark_threshold
            retouch_too_bright = retouch_mean > self.max_bri
            retouch_invalid = retouch_nonfinite | retouch_too_dark | retouch_too_bright

            if self.args.normalize_reward:
                reward_input = detect_input_loss_raw.detach() * self.cfg.detect_loss_weight
                reward_retouch = detect_retouch_loss_raw.detach() * self.cfg.detect_loss_weight
                reward_delta = (reward_input - reward_retouch) / (reward_input.abs() + float(self.args.reward_norm_eps))
            else:
                reward_delta = detect_input_loss.detach() - detect_retouch_loss

            reward = (self.cfg.all_reward + (1 - self.cfg.all_reward) * stopped) * \
                     reward_delta * self.cfg.critic_logit_multiplier
            # print("reward.shape", reward.shape, detect_input_loss.shape, detect_retouch_loss.shape)
            if self.cfg.use_penalty:
                reward -= penalty
            # print('new_states_slice', new_states)
            # print('new_states_slice', new_states[:, STATE_REWARD_DIM:STATE_REWARD_DIM + 1])
            # print('detect_retouch_loss shape', detect_retouch_loss.shape) [N, 1]

            # --- Classic actor-critic (A2C-style) ---
            # Critic target uses bootstrapped V(s') but MUST be detached to avoid "value hacking".
            reward_detached = reward.detach()
            stopped_detached = stopped.detach()
            with torch.no_grad():
                if self.cfg.use_TD:
                    new_value_target = self.value(
                        retouch.detach(), new_states.detach(), ir=(ir_after.detach() if ir_after is not None else None)
                    )
                    clear_final = torch.gt(
                        new_states[:, STATE_STEP_DIM:STATE_STEP_DIM + 1].detach(),
                        self.cfg.maximum_trajectory_length,
                    ).float()
                    new_value_target = new_value_target * (1.0 - clear_final)
                    if self.args.use_truncated:
                        # Treat invalid retouch (too dark/bright or NaN/Inf) as terminal: don't bootstrap V(s').
                        truncated = retouch_invalid.detach()
                        bootstrap_mask = (1.0 - stopped_detached) * (1.0 - truncated.float())
                        q_target = reward_detached + bootstrap_mask * self.cfg.discount_factor * new_value_target
                    else:
                        q_target = reward_detached + (1.0 - stopped_detached) * self.cfg.discount_factor * new_value_target
                else:
                    q_target = reward_detached

            old_value = self.value(imgs, states, ir=ir_before)
            advantage = q_target - old_value
            value_loss = torch.mean(advantage ** 2)
            policy_advantage = advantage.detach()
            if self.args.normalize_advantage:
                policy_advantage = (policy_advantage - policy_advantage.mean()) / (
                    policy_advantage.std(unbiased=False) + float(self.args.adv_norm_eps)
                )
            policy_loss = -torch.mean(surrogate * policy_advantage)

            # Train ISP parameter regressors via differentiable task loss (keeps policy update "classic").
            param_loss = torch.mean(detect_retouch_loss) * float(self.cfg.parameter_lr_mul)
            agent_loss = policy_loss + param_loss

            if iter % self.cfg.summary_freq == 0:
                try:
                    self.writer.add_scalar('agent_loss', agent_loss, global_step=iter)
                    self.writer.add_scalar('policy_loss', policy_loss, global_step=iter)
                    self.writer.add_scalar('param_loss', param_loss, global_step=iter)
                    self.writer.add_scalar('value_loss', value_loss, global_step=iter)
                    self.writer.add_scalar('detect_loss', detect_retouch_loss.mean(), global_step=iter)
                    self.writer.add_scalar('detect_input_loss_raw', detect_input_loss_raw.mean(), global_step=iter)
                    self.writer.add_scalar('detect_retouch_loss_raw', detect_retouch_loss_raw.mean(), global_step=iter)

                    self.writer.add_scalar('reward/mean', reward_detached.mean(), global_step=iter)
                    self.writer.add_scalar('reward/std', reward_detached.std(unbiased=False), global_step=iter)
                    self.writer.add_scalar('penalty/mean', penalty.detach().mean(), global_step=iter)
                    self.writer.add_scalar('surrogate/mean', surrogate.detach().mean(), global_step=iter)
                    self.writer.add_scalar('advantage/mean', advantage.detach().mean(), global_step=iter)
                    self.writer.add_scalar('advantage/std', advantage.detach().std(unbiased=False), global_step=iter)
                    self.writer.add_scalar('value/old_mean', old_value.detach().mean(), global_step=iter)
                    if self.cfg.use_TD:
                        self.writer.add_scalar('value/new_target_mean', new_value_target.detach().mean(), global_step=iter)

                    self.writer.add_scalar('input/mean', input_mean.mean(), global_step=iter)
                    self.writer.add_scalar('retouch/dark_threshold_mean', dark_threshold.mean(), global_step=iter)
                    self.writer.add_scalar('retouch/mean', retouch_mean.mean(), global_step=iter)
                    self.writer.add_scalar('retouch/mean_min', retouch_mean.min(), global_step=iter)
                    self.writer.add_scalar('retouch/mean_max', retouch_mean.max(), global_step=iter)
                    self.writer.add_scalar('retouch/too_dark_ratio', retouch_too_dark.float().mean(), global_step=iter)
                    self.writer.add_scalar('retouch/too_bright_ratio', retouch_too_bright.float().mean(), global_step=iter)
                    self.writer.add_scalar('retouch/nonfinite_ratio', retouch_nonfinite.float().mean(), global_step=iter)
                    self.writer.add_scalar('retouch/invalid_ratio', retouch_invalid.float().mean(), global_step=iter)
                    self.writer.add_histogram('retouch/mean_hist', retouch_mean.detach().cpu().squeeze(-1), global_step=iter)

                    selected_filter = agent_debug_out.get('selected_filter', None)
                    if selected_filter is not None:
                        selected_filter = selected_filter.detach().to(torch.int64)
                        counts = torch.bincount(selected_filter, minlength=len(self.filter_name)).float()
                        total = max(int(selected_filter.numel()), 1)
                        for i, name in enumerate(self.filter_name):
                            self.writer.add_scalar(f'filter_select/{name}', (counts[i] / total).item(), global_step=iter)
                        self.writer.add_histogram('filter_select/id', selected_filter.detach().cpu(), global_step=iter)

                    self.writer.add_images('input', torch.clip(imgs[:self.cfg.show_img_num, ...], 0.0, 1.0), global_step=iter, dataformats="NCHW")
                    if ir_before is not None:
                        ir_show = ir_before[:self.cfg.show_img_num, ...]
                        if ir_show.shape[1] == 1:
                            ir_show = ir_show.repeat(1, 3, 1, 1)
                        self.writer.add_images('ir/before', torch.clip(ir_show, 0.0, 1.0), global_step=iter, dataformats="NCHW")
                    if ir_after is not None:
                        ir_show = ir_after[:self.cfg.show_img_num, ...]
                        if ir_show.shape[1] == 1:
                            ir_show = ir_show.repeat(1, 3, 1, 1)
                        self.writer.add_images('ir/after', torch.clip(ir_show, 0.0, 1.0), global_step=iter, dataformats="NCHW")
                    # self.writer.add_images('retouch', torch.clip(retouch[:self.cfg.show_img_num, ...], 0.0, 1.0), global_step=iter, dataformats="NCHW")
                except Exception as e:
                    print("write log error!")
                # get the selected filter name
                select_filter_name = []
                filter_id = list(agent_debug_out['selected_filter'].detach().cpu().numpy())
                for id_ in filter_id:
                    select_filter_name.append(self.filter_name[id_])
                out_image = torch.clip(retouch, 0.0, 1.0).detach().cpu().numpy()
                n, c, h, w = out_image.shape
                # out_image_res = np.zeros((n, h, w, c), dtype=np.float32)
                out_image_res = []
                for b in range(n):
                    tmp = np.transpose(out_image[b, ...], (1, 2, 0)).astype(np.float32).copy()
                    tmp = cv2.putText(tmp, select_filter_name[b], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)
                    # out_image_res[b] = tmp
                    out_image_res.append(np.array(tmp))
                    # cv2.imshow('rgb', tmp)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
                try:
                    # self.writer.add_images('retouch', out_image_res[:cfg.show_img_num, ...], global_step=iter, dataformats="NHWC")
                    self.writer.add_images('retouch', np.array(out_image_res[:self.cfg.show_img_num]), global_step=iter, dataformats="NHWC")
                except Exception as e:
                    print("write log error!")
                # print(old_value, new_value)

            # Backward (separate critic/actor updates to avoid accidental gradient mixing)
            value_optimizer.zero_grad()
            value_loss.backward()
            value_grad_norm = torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
            if iter % self.cfg.summary_freq == 0:
                self.writer.add_scalar('grad_norm/value', float(value_grad_norm), global_step=iter)
            value_optimizer.step()
            value_scheduler.step()

            agent_optimizer.zero_grad()
            agent_loss.backward()
            agent_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
            if iter % self.cfg.summary_freq == 0:
                self.writer.add_scalar('grad_norm/agent', float(agent_grad_norm), global_step=iter)
            agent_optimizer.step()
            agent_scheduler.step()

            mloss_agent = (mloss_agent * iter + agent_loss.item()) / (iter + 1)  # update mean losses
            mloss_value = (mloss_value * iter + value_loss.item()) / (iter + 1)  # update mean losses
            mloss_detect = (mloss_detect * iter + detect_retouch_loss_items.cpu().numpy()) / (iter + 1)  # update mean losses
            if iter % self.cfg.print_freq == 0:
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                      ('%11s' + '%8s,') % (f'{iter}/{self.cfg.max_iter_step - 1}', mem),
                      f"agent loss: {mloss_agent:.4f},",
                      f"value loss: {mloss_value:.4f},",
                      f"box_loss: {mloss_detect[0]:.4f}, obj_loss: {mloss_detect[1]:.4f}, cls_loss: {mloss_detect[2]:.4f},",
                      f"detect_retouch_loss: {detect_retouch_loss.mean().item():.2f},",
                      f"instances: {targets.shape[0]:2d},",
                      f"agent lr: {agent_scheduler.get_lr()[0]:.4e},",
                      f"value lr: {value_scheduler.get_lr()[0]:.4e},",
                      f"penalty: {penalty.mean().item():.4e}",
                      f"reward: {reward.mean().item():.4e}",
                )
                self.train_loader.debug()
            # callbacks.run('on_train_batch_end', self.a, ni, imgs, targets, paths, list(mloss))

            # update data pool
            fill_pool_triggered = bool((
                torch.isnan(retouch).any()
                | torch.isinf(retouch).any()
                | retouch_invalid.any()
            ).item())
            if iter % self.cfg.summary_freq == 0:
                self.writer.add_scalar('replay/fill_pool_triggered', float(fill_pool_triggered), global_step=iter)

            if fill_pool_triggered:
                print(
                    "retouch invalid -> refill replay",
                    f"retouch_mean={float(retouch_mean.mean().detach().cpu()):.6f}",
                    f"dark_thr={float(dark_threshold.mean().detach().cpu()):.6f}",
                    f"too_dark_ratio={float(retouch_too_dark.float().mean().detach().cpu()):.3f}",
                    f"too_bright_ratio={float(retouch_too_bright.float().mean().detach().cpu()):.3f}",
                    f"nonfinite_ratio={float(retouch_nonfinite.float().mean().detach().cpu()):.3f}",
                )
                self.train_loader.fill_pool()
            else:
                ir_np = None
                if ir_before is not None:
                    ir_store = ir_after if (self.ir_branch_mode == "dual" and ir_after is not None) else ir_before
                    ir_np = ir_store.detach().cpu().numpy()
                self.train_loader.replace_memory(
                    self.train_loader.images_and_states_to_records(
                        retouch.detach().cpu().numpy(), feed_dict['label'], feed_dict['path'], feed_dict['shape'],
                        new_states.detach().cpu().numpy(), ir_images=ir_np))
            # validate
            if iter % self.cfg.val_freq == 0:
                self.agent.eval()
                self.yolo_model.eval()
                feed_dict = self.val_loader
                # "im", "label", "path", "shape", "state", "z": numpy
                imgs, targets, paths, shapes, states = create_input_tensor(
                    (feed_dict['im'], feed_dict['label'], feed_dict['path'], feed_dict['shape'], feed_dict['state']))
                ir_imgs_val = None
                if isinstance(imgs, (tuple, list)) and len(imgs) == 2:
                    imgs, ir_imgs_val = imgs
                for b in range(imgs.shape[0]):
                    masks = []
                    decisions = []
                    operations = []
                    debug_info_list = []
                    retouch_img_trajs = []
                    retouch = imgs[b].unsqueeze(0).to(self.device)
                    extra_val = None
                    ir_raw = None
                    ir_used = None
                    ir_raw_trajs = []
                    ir_used_trajs = []
                    if ir_imgs_val is not None:
                        ir_raw = ir_imgs_val[b].unsqueeze(0).to(self.device).float()
                        if self.ir_branch_mode == "baseline" and self.thermal_isp is not None:
                            ir_used, _ = self.thermal_isp(ir_raw)
                        else:
                            ir_used = ir_raw
                        extra_val = {"ir": ir_used}
                        ir_raw_vis = ir_raw[0].detach().cpu().numpy()
                        if ir_raw_vis.shape[0] == 1:
                            ir_raw_vis = np.repeat(ir_raw_vis, 3, axis=0)
                        ir_used_vis = ir_used[0].detach().cpu().numpy()
                        if ir_used_vis.shape[0] == 1:
                            ir_used_vis = np.repeat(ir_used_vis, 3, axis=0)
                        ir_raw_trajs.append(np.transpose(ir_raw_vis, (1, 2, 0)))
                        ir_used_trajs.append(np.transpose(ir_used_vis, (1, 2, 0)))
                    retouch_img_trajs.append(np.transpose(retouch[0].detach().cpu().numpy(), (1, 2, 0)))
                    noises = torch.from_numpy(np.array([self.train_loader.get_noise(1) for _ in range(self.cfg.test_steps)])).to(self.device)
                    states = torch.from_numpy(self.train_loader.get_initial_states(1)).to(self.device)
                    for i in range(self.cfg.test_steps):
                        (retouch, new_states, _, _), debug_info, generator_debugger = self.agent(
                            (retouch.float(), noises[i], states, extra_val), 1.0
                        )
                        retouch_img_trajs.append(np.transpose(retouch[0].detach().cpu().numpy(), (1, 2, 0)))
                        if self.ir_branch_mode == "dual" and ir_raw is not None and isinstance(debug_info, dict) and debug_info.get("ir", None) is not None:
                            ir_used = debug_info["ir"]
                            extra_val = {"ir": ir_used}
                        if ir_raw is not None and ir_used is not None:
                            ir_raw_vis = ir_raw[0].detach().cpu().numpy()
                            if ir_raw_vis.shape[0] == 1:
                                ir_raw_vis = np.repeat(ir_raw_vis, 3, axis=0)
                            ir_used_vis = ir_used[0].detach().cpu().numpy()
                            if ir_used_vis.shape[0] == 1:
                                ir_used_vis = np.repeat(ir_used_vis, 3, axis=0)
                            ir_raw_trajs.append(np.transpose(ir_raw_vis, (1, 2, 0)))
                            ir_used_trajs.append(np.transpose(ir_used_vis, (1, 2, 0)))
                        states = new_states

                        debug_info_list.append(debug_info)
                        debug_plots = generator_debugger(debug_info, combined=False)
                        decisions.append(debug_plots[0])
                        operations.append(debug_plots[1])
                        masks.append(debug_plots[2])

                        save_img(retouch, paths[b], self.image_dir, f"{iter}_{i}")
                        if states[0][STATE_STOPPED_DIM] > 0:
                            break
                    padding = 4
                    patch = 64
                    grid = patch + padding
                    steps = len(retouch_img_trajs)

                    rows = 6 if ir_raw_trajs else 4
                    fused = np.ones(shape=(grid * rows, grid * steps, 3), dtype=np.float32)

                    for i in range(len(retouch_img_trajs)):
                        sx = grid * i
                        sy = 0
                        fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
                            retouch_img_trajs[i],
                            dsize=(patch, patch),
                            interpolation=cv2.INTER_NEAREST)

                    if ir_raw_trajs and ir_used_trajs:
                        for i in range(len(retouch_img_trajs)):
                            sx = grid * i
                            sy = grid
                            fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
                                ir_raw_trajs[i],
                                dsize=(patch, patch),
                                interpolation=cv2.INTER_NEAREST)
                            sy = grid * 2
                            fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
                                ir_used_trajs[i],
                                dsize=(patch, patch),
                                interpolation=cv2.INTER_NEAREST)

                    for i in range(len(retouch_img_trajs) - 1):
                        sx = grid * i + grid // 2
                        sy = grid * (3 if ir_raw_trajs else 1)
                        fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
                            decisions[i],
                            dsize=(patch, patch),
                            interpolation=cv2.INTER_NEAREST)
                        sy = grid * (4 if ir_raw_trajs else 2) - padding // 2
                        fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
                            operations[i],
                            dsize=(patch, patch),
                            interpolation=cv2.INTER_NEAREST)
                        sy = grid * (5 if ir_raw_trajs else 3) - padding
                        fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
                            masks[i], dsize=(patch, patch), interpolation=cv2.INTER_NEAREST)

                    self.writer.add_image(f'val_{b}', fused, global_step=iter, dataformats="HWC")
                    # Save steps
                    save_img(fused, paths[b], self.image_dir, f"{iter}_steps", format="HWC")

                    # preds = self.yolo_model(retouch)
                    # # NMS
                    # targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
                    # lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
                    # preds = non_max_suppression(preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=False,
                    #                             max_det=max_det)
                    # # Metrics
                    # for si, pred in enumerate(preds):
                    #     labels = targets[targets[:, 0] == si, 1:]
                    #     nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
                    #     path, shape = Path(paths[si]), shapes[si][0]
                    #     correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
                    #     seen += 1
                    #
                    #     predn = pred.clone()
                    #     scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
                    # # Plot images
                    # if plots and batch_i < 3:
                    #     plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
                    #     plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg',
                    #                 names)  # pred

            if iter % self.cfg.save_model_freq == 0:
                self.agent.eval()
                self.value.eval()
                # Save model
                ckpt = {
                    'iter': iter,
                    'agent_model': self.agent.state_dict(),
                    'value_model': self.value.state_dict(),
                    'thermal_model': (self.thermal_isp.state_dict() if self.thermal_isp is not None else None),
                    # 'agent_scheduler': agent_scheduler.state_dict(),
                    # 'value_scheduler': value_scheduler.state_dict(),
                    'agent_optimizer': agent_optimizer.state_dict(),
                    'value_optimizer': value_optimizer.state_dict(),
                }
                # Save last, best and delete
                torch.save(ckpt, os.path.join(self.ckpt_dir, f'DynamicISP_iter_{iter}.pth'))
                del ckpt
        torch.cuda.empty_cache()

    def val(self, batch_size=1, model_weights=None, steps=5):
        # For fixed_postprocess, CLI --steps means "post" steps (excluding the forced denoise-first).
        if getattr(self.args, "isp_mode", "rl") == "fixed_postprocess":
            steps = int(steps) + 1

        z_type = "uniform"
        z_dim = 16 + 3 + len(self.cfg.filters)
        filters_number = len(self.cfg.filters)
        num_state_dim = 3 + len(self.cfg.filters)

        base_dir = self.args.val_save_path
        os.makedirs(base_dir, exist_ok=True)
        image_dir = os.path.join(base_dir, "val-images")
        os.makedirs(image_dir, exist_ok=True)
        for i in range(steps):
            os.makedirs(os.path.join(image_dir, "step-"+str(i)), exist_ok=True)
        os.makedirs(os.path.join(image_dir, "all-step"), exist_ok=True)
        # callbacks = Callbacks()
        # callbacks.run('on_val_start')

        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.args.workers} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.args.val_save_path)}\n"
                    f'Starting eval ...')

        self.agent.load_state_dict(torch.load(model_weights)['agent_model'])
        self.agent.eval()
        self.yolo_model.eval()

        if self.args.data_name in ("lod", ):
            self.val_loader, _ = create_dataloader_real_hr(self.val_path, self.args.imgsz, batch_size, self.gs, False,
                                                           hyp=self.hyp, cache=False, rect=False, workers=1, pad=0.0,
                                                           prefix=colorstr('val: '), add_noise=self.args.add_noise,
                                                           hr_original=self.args.hr_original,
                                                           apply_meta_wb_ccm=getattr(self.args, "apply_meta_wb_ccm", False))
        s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
        pbar = tqdm(self.val_loader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
        for batch_i, (imgs, targets, paths, shapes, imgs_hr) in enumerate(pbar):
        # self.train_loader.load()
        # feed_dict = self.train_loader.get_feed_dict_and_states(8)
        # imgs, targets, paths, shapes, states = create_input_tensor(
        #     (feed_dict['im'], feed_dict['label'], feed_dict['path'], feed_dict['shape'], feed_dict['state']))
        # for b in range(imgs.shape[0]):
            # callbacks.run('on_val_batch_start')
            masks = []
            decisions = []
            operations = []
            debug_info_list = []
            retouch_img_trajs = []
            retouch = imgs.to(self.device)
            retouch_hr = imgs_hr.to(self.device)
            # retouch = imgs[b].unsqueeze(0).to(self.device)
            retouch_img_trajs.append(np.transpose(retouch[0].detach().cpu().numpy(), (1, 2, 0)))

            nb = imgs.shape[0]
            noises = torch.from_numpy(np.array([get_noise(nb, z_type, z_dim) for _ in range(steps)])).to(self.device)
            states = torch.from_numpy(get_initial_states(nb, num_state_dim, filters_number)).to(self.device)
            for i in range(steps):
                (retouch, new_states, retouch_hr), debug_info, generator_debugger = self.agent((retouch.float(), noises[i], states), 1.0, retouch_hr.float())
                retouch_img_trajs.append(np.transpose(retouch[0].detach().cpu().numpy(), (1, 2, 0)))
                states = new_states

                debug_info_list.append(debug_info)
                debug_plots = generator_debugger(debug_info, combined=False)
                decisions.append(debug_plots[0])
                operations.append(debug_plots[1])
                masks.append(debug_plots[2])

                # save_img(retouch_hr, paths[0], image_dir, f"{i}")
                save_img(retouch_hr, paths[0], os.path.join(image_dir, "step-" +str(i)), None, "CHW", False)
                if states[0][STATE_STOPPED_DIM] > 0:
                    break
            padding = 4
            patch = 64
            grid = patch + padding
            steps = len(retouch_img_trajs)

            fused = np.ones(shape=(grid * 4, grid * steps, 3), dtype=np.float32)

            for i in range(len(retouch_img_trajs)):
                sx = grid * i
                sy = 0
                fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
                    retouch_img_trajs[i],
                    dsize=(patch, patch),
                    interpolation=cv2.INTER_NEAREST)

            for i in range(len(retouch_img_trajs) - 1):
                sx = grid * i + grid // 2
                sy = grid
                fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
                    decisions[i],
                    dsize=(patch, patch),
                    interpolation=cv2.INTER_NEAREST)
                sy = grid * 2 - padding // 2
                fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
                    operations[i],
                    dsize=(patch, patch),
                    interpolation=cv2.INTER_NEAREST)
                sy = grid * 3 - padding
                fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
                    masks[i], dsize=(patch, patch), interpolation=cv2.INTER_NEAREST)

            # Save steps
            # save_img(fused, paths[0], image_dir, f"steps", format="HWC")
            save_img(fused, paths[0], os.path.join(image_dir, "all-step"), None, "HWC", False)

            # preds = self.yolo_model(retouch)
            # # NMS
            # targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
            # lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            # preds = non_max_suppression(preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=False,
            #                             max_det=max_det)
            # # Metrics
            # for si, pred in enumerate(preds):
            #     labels = targets[targets[:, 0] == si, 1:]
            #     nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            #     path, shape = Path(paths[si]), shapes[si][0]
            #     correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            #     seen += 1
            #
            #     predn = pred.clone()
            #     scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
            # # Plot images
            # if plots and batch_i < 3:
            #     plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            #     plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg',
            #                 names)  # pred
        torch.cuda.empty_cache()


def _fixed_filter_cls(name: str):
    from isp.filters import ToneFilter, ContrastFilter, SharpenFilter, SaturationPlusFilter, ExposureFilter

    key = (name or "").strip().lower()
    mapping = {
        "tone": ToneFilter,
        "contrast": ContrastFilter,
        "sharpen": SharpenFilter,
        "saturation": SaturationPlusFilter,
        "exposure": ExposureFilter,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported --fixed_learn_filter={name!r}, choose from {sorted(mapping.keys())}")
    return mapping[key]


def run_fixed_isp(args):
    """Deterministic ISP training: fixed WB/CCM upstream + learn denoise + one module + gamma(last)."""
    try:
        cfg = importlib.import_module(f'{args.cfg}').cfg
    except Exception as e:
        raise RuntimeError(f"Failed to import cfg module {args.cfg!r}: {e}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = os.path.join('experiments', args.save_path + '-fixedisp')
    os.makedirs(base_dir, exist_ok=True)
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(base_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    Tee(os.path.join(log_dir, 'log.txt'))
    writer = SummaryWriter(log_dir)
    LOGGER.info(f"[fixed] outputs: {base_dir}")
    LOGGER.info(f"[fixed] logs: {log_dir}")
    LOGGER.info(f"[fixed] ckpt: {ckpt_dir}")
    LOGGER.info("[fixed] note: fixed mode does not save preview images by default; use --fixed_debug_dump true if needed.")

    hyp = args.hyp
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    data_dict = check_dataset(args.data_cfg)
    nc = int(data_dict['nc'])  # number of classes
    train_path = data_dict['train']

    # Load Pretrained YOLO
    with torch_distributed_zero_first(LOCAL_RANK):
        weights = attempt_download(args.weights)  # download if not found locally
    ckpt = torch.load(weights, map_location='cpu')
    yolo_model = Model(args.yolo_cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    exclude = ['anchor'] if (args.yolo_cfg or hyp.get('anchors')) else []
    csd = ckpt['model'].float().state_dict()
    csd = intersect_dicts(csd, yolo_model.state_dict(), exclude=exclude)
    yolo_model.load_state_dict(csd, strict=False)
    LOGGER.info(f'Transferred {len(csd)}/{len(yolo_model.state_dict())} items from {weights}')

    # Match DynamicISP attribute wiring expected by ComputeLoss()
    nl = yolo_model.model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl
    hyp['cls'] *= nc / 80 * 3 / nl
    hyp['obj'] *= (args.imgsz / 640) ** 2 * 3 / nl
    hyp['label_smoothing'] = 0.0
    yolo_model.nc = nc
    yolo_model.hyp = hyp
    yolo_model.names = data_dict['names']

    # Freeze YOLO
    yolo_model.train()
    for p in yolo_model.parameters():
        p.requires_grad = False
    for m in yolo_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    gs = max(int(yolo_model.stride.max()), 32)
    args.imgsz = check_img_size(args.imgsz, gs, floor=gs * 2)

    if args.fixed_input == "srgb":
        train_loader, dataset = create_dataloader_real(
            train_path,
            args.imgsz,
            args.batch_size,
            gs,
            single_cls=False,
            hyp=hyp,
            augment=False,
            cache=False,
            pad=0.0,
            rect=False,
            image_weights=False,
            prefix=colorstr('train: '),
            limit=-1,
            workers=args.workers,
        )
    else:
        train_loader, dataset = create_dataloader(
            train_path,
            args.imgsz,
            args.batch_size,
            gs,
            single_cls=False,
            hyp=hyp,
            augment=False,
            cache=False,
            pad=0.0,
            rect=False,
            image_weights=False,
            prefix=colorstr('train: '),
            limit=-1,
            add_noise=args.add_noise,
            brightness_range=args.bri_range,
            noise_level=args.noise_level,
            use_linear=args.use_linear,
            apply_meta_wb_ccm=args.apply_meta_wb_ccm,
            workers=args.workers,
        )

    if getattr(args, "fixed_debug_dump", False):
        debug_dir = os.path.join(base_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        try:
            # Note: common.raw_reader appends SeAFusion/ into sys.path, which makes the top-level
            # module name "SeAFusion" resolve to SeAFusion/SeAFusion.py (a module, not a package).
            # Import via common.raw_reader to avoid "SeAFusion is not a package" errors.
            from common.raw_reader import process_hq_dng_file as seafusion_process_hq_dng_file

            sample_path = dataset.im_files[0]
            t0 = seafusion_process_hq_dng_file(sample_path, output_channels=3, apply_wb_ccm=False)
            t1 = seafusion_process_hq_dng_file(sample_path, output_channels=3, apply_wb_ccm=bool(args.apply_meta_wb_ccm))
            if t0 is not None and t1 is not None:
                a = t0.detach().cpu().numpy()[0].transpose(1, 2, 0)
                b = t1.detach().cpu().numpy()[0].transpose(1, 2, 0)
                diff = float(np.mean(np.abs(a - b)))
                LOGGER.info(f"[fixed][debug] apply_wb_ccm diff(mean abs)={diff:.6f} path={sample_path}")
                # Display gamma for visualization only
                a_disp = np.clip(a, 0.0, 1.0) ** (1.0 / 2.2)
                b_disp = np.clip(b, 0.0, 1.0) ** (1.0 / 2.2)
                cv2.imwrite(os.path.join(debug_dir, "dng_nowbccm_gamma22.png"),
                            cv2.cvtColor((a_disp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(debug_dir, "dng_wbccm_gamma22.png"),
                            cv2.cvtColor((b_disp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        except Exception as e:
            LOGGER.warning(f"[fixed][debug] failed to dump WB/CCM debug images: {e}")

    learned_cls = _fixed_filter_cls(args.fixed_learn_filter)
    isp = FixedISPPipeline(cfg, learned_cls).to(device)
    optimizer = torch.optim.Adam(isp.parameters(), lr=args.lr)
    try:
        yolo_model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    except Exception:
        pass
    compute_loss = ComputeLoss(yolo_model)

    # Number of optimizer steps per "epoch" (ceil to keep at least 1 batch when n < batch_size)
    steps_per_epoch = max(1, int(math.ceil(len(dataset) / max(1, args.batch_size))))
    max_steps = max(1, int(args.epochs) * steps_per_epoch)
    LOGGER.info(f"Fixed ISP training steps: {max_steps} (epochs={args.epochs}, steps/epoch={steps_per_epoch})")

    loader_iter = iter(train_loader)
    for step in range(max_steps):
        try:
            imgs, targets, paths, shapes = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            imgs, targets, paths, shapes = next(loader_iter)

        isp.train()
        imgs = imgs.to(device, non_blocking=True).float()
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        retouch, dbg = isp(imgs)
        preds = yolo_model(retouch)
        loss, loss_items = compute_loss(preds, targets)
        loss.backward()
        optimizer.step()

        if step % cfg.print_freq == 0:
            LOGGER.info(
                f"[fixed] step {step}/{max_steps} loss={loss.item():.4f} "
                f"box={loss_items[0].item():.4f} obj={loss_items[1].item():.4f} cls={loss_items[2].item():.4f}"
            )

        if step % cfg.summary_freq == 0:
            writer.add_scalar("fixed/loss_total", loss.item(), step)
            writer.add_scalar("fixed/loss_box", loss_items[0].item(), step)
            writer.add_scalar("fixed/loss_obj", loss_items[1].item(), step)
            writer.add_scalar("fixed/loss_cls", loss_items[2].item(), step)
            try:
                writer.add_scalar("fixed/gamma", float(torch.mean(dbg["gamma"]).detach().cpu()), step)
            except Exception:
                pass

        if step % cfg.save_model_freq == 0 and step > 0:
            save_path = os.path.join(ckpt_dir, f"FixedISP_step_{step}.pth")
            torch.save({"step": step, "isp": isp.state_dict(), "args": vars(args)}, save_path)

    save_path = os.path.join(ckpt_dir, "FixedISP_final.pth")
    torch.save({"step": max_steps, "isp": isp.state_dict(), "args": vars(args)}, save_path)
    writer.close()


#PYTHONPATH=.. python train.py --data_cfg yolov3/data/lod.yaml --task train_val --data_name lod --weights /root/autodl-tmp/yolov3.pt --yolo_cfg yolov3/models/yolov3.yaml --hyp yolov3/data/hyps/hyp.scratch-low.yaml  --resume experiments/lod-adaptiveisp/ckpt/DynamicISP_iter_41000.pth
if __name__ == "__main__":
    import argparse

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return True
        val = str(v).strip().lower()
        if val in ("1", "true", "t", "yes", "y", "on"):
            return True
        if val in ("0", "false", "f", "no", "n", "off"):
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

    def add_bool_flag(parser, name, default, help):
        parser.add_argument(name, type=str2bool, nargs='?', const=True, default=default, help=help)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--isp_mode",
        type=str,
        default="rl",
        choices=["rl", "fixed", "fixed_postprocess"],
        help="rl: original DynamicISP (RL). fixed: fixed-order ISP (learn params only). "
             "fixed_postprocess: RL mode with fixed meta WB/CCM + optional denoise-first, then RL-selected post pipeline.",
    )
    parser.add_argument("--task", type=str, default='train_val', help="train, train and val, val")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--epochs", type=int, default=800, help="epochs")
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--scheduler_step_size", type=int, default=20, help="scheduler_step_size")
    parser.add_argument("--scheduler_lr_gamma", type=float, default=0.5, help="scheduler_lr_gamma")
    parser.add_argument("--imgsz", type=int, default=512, help="image size")
    parser.add_argument("--hr_original", action="store_true", default=False,
                        help="in val(), run ISP high_res branch on original-resolution images for saving")
    parser.add_argument("--workers", type=int, default=4, help="workers")
    
    parser.add_argument('--weights', type=str, default='../../pretrained/yolov3.pt', help='yolov3 pretrained path')
    parser.add_argument('--yolo_cfg', type=str, default='yolov3/models/yolov3.yaml', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='yolov3/data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')

    parser.add_argument("--save_path", type=str, default='adaptiveisp', help="save path at experiments/save_path/")
    parser.add_argument("--data_name", type=str, default='lod', choices=['lod', 'vif'], help="train data: lod or paired vif")
    parser.add_argument('--data_cfg', type=str, default='data/lod.yaml', help='dataset.yaml path (relative to yolov3)')
    parser.add_argument("--vi_dir_name", type=str, default="vi", help="visible dir name for paired vif path mapping")
    parser.add_argument("--ir_dir_name", type=str, default="ir", help="infrared dir name for paired vif path mapping")
    parser.add_argument("--ir_root", type=str, default=None, help="optional infrared root dir (overrides vi/ir dir mapping)")
    add_bool_flag(parser, "--ir_use_y16", default=True, help="thermal raw: use Y16 (else use YUV part if available)")
    parser.add_argument("--ir_width", type=int, default=None, help="thermal raw: override width (optional)")
    parser.add_argument("--ir_height", type=int, default=None, help="thermal raw: override height (optional)")
    parser.add_argument("--ir_branch_mode", type=str, default="baseline", choices=["baseline", "dual"],
                        help="baseline: fixed thermal ISP (GD+LT+AGC) before fusion; "
                             "dual: action space includes IR filters and IFCNN fuses after every step")
    add_bool_flag(parser, "--add_noise", default=False, help="add_noise")
    parser.add_argument("--use_linear", action='store_true', default=False, help="use linear noise distribution")
    parser.add_argument("--bri_range", type=float, default=None, nargs='*', help="brightness range, (low, high), 0.0~1.0")
    parser.add_argument("--noise_level", type=float, default=None, help="noise_level, 0.001~0.012")
    parser.add_argument("--fixed_input", type=str, default="raw", choices=["raw", "srgb"],
                        help="fixed mode input: raw=LoadImagesAndLabelsRAW, srgb=LoadImagesAndLabelsNormalize")
    parser.add_argument("--fixed_learn_filter", type=str, default="tone",
                        choices=["tone", "contrast", "sharpen", "saturation", "exposure"],
                        help="fixed mode: learn exactly one extra module besides denoise+gamma")
    add_bool_flag(parser, "--fixed_debug_dump", default=False, help="fixed mode: dump WB/CCM debug images")
    add_bool_flag(parser, "--apply_meta_wb_ccm", default=True, help="DNG only: apply metadata WB+CCM as fixed steps")
    add_bool_flag(parser, "--fixed_postprocess_force_gamma", default=True,
                  help="fixed_postprocess: force GammaFilter as the last step")
    add_bool_flag(parser, "--force_nlm_first_in_fixed_postprocess_mode", default=False,
                  help="fixed_postprocess: force NLM as the first step (step 0)")
    parser.add_argument("--gamma_init", type=float, default=2.2,
                        help="gamma init value (display gamma, e.g., 2.2); <=0 disables bias")
    parser.add_argument("--nlm_limit", type=float, default=-1.0,
                        help="limit NLM strength by scaling its predicted parameter to [0, nlm_limit]; <0 disables")
    parser.add_argument("--nlm_init", type=float, default=-1.0,
                        help="initialize NLM strength (0<nlm_init<1); <0 disables")

    add_bool_flag(parser, "--use_truncated", default=False, help="use_truncated")
    add_bool_flag(parser, "--masking", default=None, help="enable per-operator masking (also affects mask visualization)")
    add_bool_flag(parser, "--relative_brightness", default=False, help="use relative (input-conditioned) dark threshold for invalid retouch")
    parser.add_argument("--retouch_dark_ratio", type=float, default=0.2, help="relative dark threshold: min(abs, input_mean * ratio)")
    parser.add_argument("--retouch_dark_abs", type=float, default=0.01, help="absolute cap for retouch dark threshold")
    add_bool_flag(parser, "--normalize_reward", default=False, help="normalize reward by input detection loss magnitude")
    parser.add_argument("--reward_norm_eps", type=float, default=1e-6, help="epsilon for reward normalization")
    add_bool_flag(parser, "--normalize_advantage", default=False, help="standardize advantage for policy loss")
    parser.add_argument("--adv_norm_eps", type=float, default=1e-6, help="epsilon for advantage standardization")
    parser.add_argument("--runtime_penalty", action='store_true', default=False, help="use runtime penalty")
    parser.add_argument("--runtime_penalty_lambda", type=float, default=0.01, help="use runtime penalty lambda")
    parser.add_argument('--resume', type=str, default=None, help='resume model weights')

    parser.add_argument('--model_weights', type=str, default='experiments/', help='isp model weight')
    parser.add_argument("--val_save_path", type=str, default='experiments/adaptiveisp')
    parser.add_argument("--steps", type=int, default=5, help="steps")
    parser.add_argument("--cfg", type=str, default="config", help="config py file")
    parser.add_argument("--force_filter_name", type=str, default=None,
                        help="force selecting this filter short name (e.g., IF); default uses cfg.force_filter_name")
    parser.add_argument("--force_filter_step", type=int, default=None,
                        help="0-indexed step to force filter selection; -1 disables; default uses cfg.force_filter_step")
    parser.add_argument("--ifcnn_weights", type=str, default=None, help="IFCNN weights path (for IFCNNFusionFilter)")
    add_bool_flag(parser, "--ifcnn_trainable", default=None, help="fine-tune IFCNN weights during training")
    parser.add_argument("--ifcnn_fuse_scheme", type=int, default=None, help="IFCNN fuse scheme: 0=MAX, 1=SUM, 2=MEAN")

    args = parser.parse_args()
    args.save_path = args.data_name + '-' + args.save_path
    if args.data_name in ("lod", ):
        args.add_noise = False
        args.bri_range = None
        args.use_linear = False

    if args.isp_mode == "fixed":
        if args.task in ("train", "train_val"):
            run_fixed_isp(args)
        else:
            raise ValueError(f"fixed mode only supports --task train/train_val for now, got {args.task!r}")
    else:
        Task = DynamicISP(args, args.task)
        if args.task == "train" or args.task == "train_val":
            Task.train()
        elif args.task == "val":
            Task.val(model_weights=args.model_weights, steps=args.steps)
