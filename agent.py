import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from util import enrich_image_input
from util import STATE_DROPOUT_BEGIN, STATE_REWARD_DIM, STATE_STEP_DIM, STATE_STOPPED_DIM
from fusion.ifcnn import build_ifcnn, ensure_3ch, imagenet_denormalize, imagenet_normalize, load_ifcnn_weights

def pdf_sample(pdf, uniform_noise):
    pdf = pdf / (torch.sum(pdf, dim=1, keepdim=True) + 1e-36)
    cdf = torch.cumsum(pdf, dim=1) - pdf
    indices = torch.sum(torch.less(cdf, uniform_noise).to(torch.int32), dim=1) - 1
    return indices

def one_hot(num_class, index):
    label = torch.zeros((num_class, *index.shape), dtype=torch.int64, device=index.device)
    for i in range(num_class):
        label[i, index == i] = 1
    label = label.permute(1, 0)
    return label


class FeatureExtractor(torch.nn.Module):
    def __init__(self, shape=(14, 64, 64), mid_channels=32, output_dim=4096, dropout_prob=0.5):
        """shape: c,h,w"""
        super(FeatureExtractor, self).__init__()
        in_channels = shape[0]
        self.output_dim = output_dim

        min_feature_map_size = 4
        assert output_dim % (min_feature_map_size ** 2) == 0, 'output dim=%d' % output_dim
        size = int(shape[2])
        size = size // 2
        channels = mid_channels
        layers = []
        layers.append(nn.Conv2d(in_channels, channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        while size > min_feature_map_size:
            in_channels = channels
            if size == min_feature_map_size * 2:
                channels = output_dim // (min_feature_map_size ** 2)
            else:
                channels *= 2
            assert size % 2 == 0
            size = size // 2
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.layers = nn.Sequential(*layers)
        self.droupout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.layers(x)
        x = torch.reshape(x, [-1, self.output_dim])
        x = self.droupout(x)
        return x


# Output: float \in [0, 1]
class Agent(nn.Module):
    def __init__(self, cfg, shape=(16, 64, 64), device='cuda', meta_ccm=None):
        super(Agent, self).__init__()
        self.cfg = cfg
        self.include_ir_in_agent = bool(cfg.get("include_ir_in_agent", False))
        self.fuse_after_each_step = bool(cfg.get("fuse_after_each_step", False))
        self.fuser = None
        self.fuser_trainable = False
        self.feature_extractor = FeatureExtractor(shape=shape, mid_channels=cfg.base_channels,
                                                  output_dim=cfg.feature_extractor_dims,
                                                  dropout_prob=1.0 - cfg.dropout_keep_prob)
        self.filters = []
        for func in self.cfg.filters:
            if func.__name__ == 'CCMFilter':
                filter = func(self.cfg, predict=True, meta_ccm=meta_ccm).to(device)
            else:
                filter = func(self.cfg, predict=True).to(device)
            self.__setattr__(filter.get_short_name(), filter)
            self.filters.append(filter)

        self.action_selection = FeatureExtractor(shape=shape, mid_channels=cfg.base_channels,
                                                 output_dim=cfg.feature_extractor_dims,
                                                 dropout_prob=1.0 - cfg.dropout_keep_prob)

        self.fc1 = nn.Linear(cfg.feature_extractor_dims, cfg.fc1_size)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(cfg.fc1_size, len(self.filters))
        self.softmax = nn.Softmax(dim=1)
        self.down_sample = nn.AdaptiveAvgPool2d((shape[1], shape[2]))
        self.runtime = torch.tensor(cfg.filters_runtime, requires_grad=False).to(device)

        if self.fuse_after_each_step:
            fuse_scheme = int(cfg.get("ifcnn_fuse_scheme", 0))
            self.fuser = build_ifcnn(fuse_scheme=fuse_scheme, resnet_pretrained=False).to(device)
            weights_path = cfg.get("ifcnn_weights", None)
            if weights_path:
                load_ifcnn_weights(self.fuser, weights_path, map_location="cpu")

            self.fuser_trainable = bool(cfg.get("ifcnn_trainable", False))
            for p in self.fuser.parameters():
                p.requires_grad = self.fuser_trainable
            if not self.fuser_trainable:
                self.fuser.eval()

        # Forced filter selection schedule: step (int) -> filter id (int).
        # Supports the legacy single (force_filter_name, force_filter_step) and an optional
        # cfg.force_filter_schedule mapping, e.g. {0: "NLM", 5: "G"}.
        self.force_filter_ids_by_step = {}
        legacy_step = cfg.get("force_filter_step", None)
        legacy_name = cfg.get("force_filter_name", None)
        if legacy_name is None:
            legacy_name = cfg.get("force_filter_short_name", None)
        if legacy_name is not None and legacy_step is not None:
            for i, f in enumerate(self.filters):
                if f.get_short_name() == str(legacy_name):
                    self.force_filter_ids_by_step[int(legacy_step)] = int(i)
                    break

        schedule = cfg.get("force_filter_schedule", None)
        if schedule:
            items = schedule.items() if isinstance(schedule, dict) else schedule
            for step, name in items:
                if name is None:
                    continue
                for i, f in enumerate(self.filters):
                    if f.get_short_name() == str(name):
                        try:
                            self.force_filter_ids_by_step[int(step)] = int(i)
                        except Exception:
                            pass
                        break

        self.disallow_after_step0_ids = []
        disallow_names = cfg.get("disallow_filter_short_names_after_step0", None)
        if disallow_names:
            disallow_set = set(disallow_names)
            for i, f in enumerate(self.filters):
                if f.get_short_name() in disallow_set:
                    self.disallow_after_step0_ids.append(int(i))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.fuse_after_each_step and self.fuser is not None and not self.fuser_trainable:
            self.fuser.eval()
        return self

    @staticmethod
    def _ir_as_gray(ir: torch.Tensor) -> torch.Tensor:
        if ir.dim() != 4:
            raise ValueError(f"Expected IR as NCHW, got shape={tuple(ir.shape)}")
        if ir.shape[1] == 1:
            return ir
        if ir.shape[1] == 3:
            return (0.27 * ir[:, 0:1] + 0.67 * ir[:, 1:2] + 0.06 * ir[:, 2:3])
        raise ValueError(f"Expected IR with 1 or 3 channels, got C={ir.shape[1]}")

    def _fuse_vi_ir(self, vi: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        if self.fuser is None:
            return vi
        vi = torch.clip(ensure_3ch(vi), 0.0, 1.0)
        ir = torch.clip(ensure_3ch(ir), 0.0, 1.0)
        vi_n = imagenet_normalize(vi)
        ir_n = imagenet_normalize(ir)
        out_n = self.fuser(vi_n, ir_n)
        out = imagenet_denormalize(out_n)
        return torch.clip(out, 0.0, 1.0)

    def forward(self, inp, progress, high_res=None, selected_filter_id=None):
        train = 1 if self.training else 0
        extra = None
        if isinstance(inp, (tuple, list)) and len(inp) == 4:
            x, z, states, extra = inp
        else:
            x, z, states = inp
        ir = None
        if isinstance(extra, dict):
            ir = extra.get("ir", None)

        selection_noise = z[:, 0:1]
        filtered_images = []
        filter_debug_info = []
        high_res_outputs = []
        ir_candidates = [] if (self.fuse_after_each_step and ir is not None) else None

        x_down = self.down_sample(x)
        if self.include_ir_in_agent and ir is not None:
            ir_down = self.down_sample(self._ir_as_gray(ir))
            x_down = torch.cat([x_down, ir_down], dim=1)
        if self.cfg.shared_feature_extractor:
            filter_features = self.feature_extractor(enrich_image_input(self.cfg, x_down, states))
        else:
            raise ValueError("current just support shared_feature_extractor")
        # filter_features.sum().backward()
        for j, filter in enumerate(self.filters):
            # print('    creating filter:', j, 'name:', str(filter.__class__), 'abbr.', filter.get_short_name())
            # print('      filter_features:', filter_features.shape)
            if ir_candidates is not None and getattr(filter, "branch", None) == "ir":
                ir_out, _, per_filter_debug_info = filter(ir, filter_features, high_res=None, extra=None)
                filtered_image_batch = x
                high_res_output = high_res if high_res is not None else None
                ir_candidates.append(ir_out)
            else:
                filtered_image_batch, high_res_output, per_filter_debug_info = filter(
                    x, filter_features, high_res=high_res, extra=extra
                )
                if ir_candidates is not None:
                    ir_candidates.append(ir)
            high_res_outputs.append(high_res_output)
            filtered_images.append(filtered_image_batch)
            filter_debug_info.append(per_filter_debug_info)
            # print('      output:', filtered_image_batch.shape)
            # filtered_image_batch.sum().backward()

        # [batch_size, #filters, H, W, C]
        # for img in filtered_images:
        #     print('img', img.shape)
        filtered_images = torch.stack(filtered_images, dim=1)
        # print('    filtered_images:', filtered_images.shape)

        # filtered_images.sum().backward()
        # action_selection
        selector_features = self.action_selection(enrich_image_input(self.cfg, x_down, states))
        # print('    selector features:', selector_features.shape)
        selector_features = self.lrelu(self.fc1(selector_features))

        # print('    selector features:', selector_features.shape)
        pdf = self.softmax(self.fc2(selector_features)) + 1e-37
        # print('    pdf_filter', pdf[:, 1:].shape)

        pdf = pdf * (1 - self.cfg.exploration) + self.cfg.exploration * 1.0 / len(self.filters)
        # pdf = tf.to_float(is_train) * tf.concat([pdf[:, :1], pdf[:, 1:] * states[:, STATE_DROPOUT_BEGIN:]], axis=1) \
        # + (1.0 - tf.to_float(is_train)) * pdf
        if self.disallow_after_step0_ids:
            step = states[:, STATE_STEP_DIM]
            not_first = step > 0.5
            if torch.any(not_first):
                pdf = pdf.clone()
                for idx in self.disallow_after_step0_ids:
                    pdf[not_first, idx] = 0.0
                row_sum = torch.sum(pdf, dim=1, keepdim=True)
                zero_rows = row_sum < 1e-12
                if torch.any(zero_rows):
                    pdf[zero_rows.squeeze(1)] = 1.0 / len(self.filters)
        pdf = pdf / (torch.sum(pdf, dim=1, keepdim=True) + 1e-30)
        # Avoid NaNs when some actions are hard-masked to probability 0 (e.g., fixed_postprocess).
        entropy = -pdf * torch.log(pdf + 1e-10)
        entropy = torch.sum(entropy, dim=1)[:, None]
        # print('    pdf:', pdf.shape)
        # print('    entropy:', entropy.shape)
        # print('    selection_noise:', selection_noise.shape)
        random_filter_id = pdf_sample(pdf, selection_noise)
        max_filter_id = torch.argmax(pdf, dim=1).to(torch.int32)
        if selected_filter_id is not None:
            selected_filter_id = torch.from_numpy(np.array([selected_filter_id] * max_filter_id.shape[0])).to(torch.int64).to(max_filter_id.device)
        else:
            selected_filter_id = (train * random_filter_id + (1 - train) * max_filter_id).to(torch.int64)

        # Optionally force specific filters at specific steps.
        if self.force_filter_ids_by_step:
            step = states[:, STATE_STEP_DIM:STATE_STEP_DIM + 1]
            for force_step in sorted(self.force_filter_ids_by_step.keys()):
                force_mask = (torch.abs(step - float(force_step)) < 1e-4).squeeze(1)
                if torch.any(force_mask):
                    forced = torch.full_like(selected_filter_id, int(self.force_filter_ids_by_step[force_step]))
                    selected_filter_id = torch.where(force_mask, forced, selected_filter_id)
        # print("selected_filter_id", selected_filter_id, random_filter_id, max_filter_id)
        # print('    selected_filter_id:', selected_filter_id.shape)

        # selected_filter_id = torch.clip(selected_filter_id, min=0, max=len(self.filters)-1)
        # filter_one_hot = F.one_hot(selected_filter_id, num_classes=len(self.filters))
        filter_one_hot = one_hot(len(self.filters), selected_filter_id)

        # print('    filter one_hot', filter_one_hot.shape, filter_one_hot)
        surrogate = torch.sum(filter_one_hot * torch.log(pdf + 1e-10), dim=1, keepdim=True)

        x = torch.sum(filtered_images * filter_one_hot[:, :, None, None, None], dim=1)
        selected_ir = None
        if ir_candidates is not None:
            selected_ir = torch.sum(torch.stack(ir_candidates, dim=1) * filter_one_hot[:, :, None, None, None], dim=1)

        # Dual-branch mode: always fuse (VIF) after the chosen branch operation.
        if self.fuse_after_each_step and selected_ir is not None:
            x = self._fuse_vi_ir(x, selected_ir)
        if high_res is not None:
            high_res_outputs = torch.stack(high_res_outputs, dim=1)
            high_res_output = torch.sum(high_res_outputs * filter_one_hot[:, :, None, None, None], dim=1)

        # only the first image will get debug_info
        debug_info = {
            'state': states,
            'selected_filter_id': selected_filter_id[0],
            'filter_debug_info': filter_debug_info,
            'pdf': pdf[0],
            'selected_filter': selected_filter_id,
        }
        if selected_ir is not None:
            debug_info["ir"] = selected_ir

        # Combined: Three in one 64x64
        #           otherwise returns pdf, detail, mask
        def debugger(debug_info, combined=True):
            size = len(self.cfg.filters)  # 8
            img = None
            images = [None for i in range(3)]
            for i, filter in enumerate(self.filters):
                selected = i == debug_info['selected_filter_id']
                if selected:
                    img = filter.visualize_mask(debug_info['filter_debug_info'][i], (64, 64)) * 0.8
            assert img is not None
            if not combined:
                # Mask
                images[2] = img.copy()
                # reset img
                img = img * 0 + 0.5

            c = 0
            for i, filter in enumerate(self.filters):
                pdf = debug_info['pdf'][i]
                if pdf < 1e-10:
                    continue
                else:
                    c += 1
                selected = i == debug_info['selected_filter_id']
                if selected:
                    filter.visualize_filter(debug_info['filter_debug_info'][i], img)
            if not combined:
                # detail
                images[1] = img.copy()
                # reset img
                img = img * 0 + 0.5
            c = 0
            for i, filter in enumerate(self.filters):
                per_col = (len(self.cfg.filters) + 1) // 2  # 4
                x = c // per_col * 30
                y = size * (c % per_col + 1)
                pdf = debug_info['pdf'][i]
                if pdf < 1e-10:
                    continue
                else:
                    c += 1
                cv2.putText(img, filter.get_short_name(), (x + 6, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.233,
                            (255, 255, 255))
                selected = i == debug_info['selected_filter_id']
                color = 1.0 if selected else 0.3
                width = int(pdf * 20)
                height = 0.35
                corners = [(x + 16, int(y + (1 - height) * size // 2)),
                           (x + 16 + width, int(y + (1 + height) * size // 2))]
                cv2.rectangle(img, (corners[0][0] - 1, corners[0][1] - 1),
                              (corners[1][0] + 1, corners[1][1] + 1), (1, 1, 1), cv2.FILLED)
                cv2.rectangle(img, corners[0], corners[1], (color, 0.3, 0.3), cv2.FILLED)
            if not combined:
                # pdf
                images[0] = img.copy()

            if combined:
                return img
            else:
                return images

        debugger.width = int(x.shape[2])
        # print('    surrogate: ', surrogate.shape)

        # Calculate new states
        new_states = [None for _ in range(STATE_DROPOUT_BEGIN + 1)]
        is_last_step = (torch.abs(states[:, STATE_STEP_DIM:STATE_STEP_DIM + 1] + 1 - self.cfg.test_steps)
                        < 1e-4).to(torch.float32)
        submitted = is_last_step

        new_states[STATE_REWARD_DIM] = submitted
        new_states[STATE_STOPPED_DIM] = submitted
        # Increment the step
        new_states[STATE_STEP_DIM] = (states[:, STATE_STEP_DIM] + 1)[:, None]

        # Update filter usage
        filter_usage = states[:, STATE_STEP_DIM + 1:]
        # print('usage v.s. onehot', filter_usage.shape, filter_one_hot.shape)
        assert len(filter_usage.shape) == len(filter_one_hot.shape)

        regular_filter_start = 0

        # Penalize submission action that is not the final action.
        early_stop_penalty = (1 - is_last_step) * submitted * self.cfg.early_stop_penalty

        usage_penalty = torch.sum(filter_usage * filter_one_hot[:, regular_filter_start:], dim=1, keepdim=True)
        new_filter_usage = torch.maximum(filter_usage, filter_one_hot[:, regular_filter_start:])
        new_states[STATE_STEP_DIM + 1] = new_filter_usage

        # print("submitted.shape, new_states[STATE_STEP_DIM].shape", submitted.shape, new_states[STATE_STEP_DIM].shape)
        new_states = torch.cat(new_states, dim=1)
        # print('new_states:', new_states.shape)

        if self.cfg.clamp:
            x = torch.clip(x, min=0.0, max=5.0)

        entropy_penalty = (1.0 - progress) * self.cfg.exploration_penalty * (-entropy + math.log(len(self.filters)))

        runtime_penalty = 0.0
        if self.cfg.filter_runtime_penalty:
            runtime_penalty = torch.sum(filter_one_hot * self.runtime, dim=1, keepdim=True)
            runtime_penalty = self.cfg.filter_runtime_penalty_lambda * runtime_penalty
            # print("entropy_penalty", entropy_penalty)
            # print("early_stop_penalty", early_stop_penalty)
            # print("runtime_penalty", runtime_penalty)

        # Will be substracted from award
        penalty = torch.mean(torch.clip(x - 1, min=0)**2, dim=(1, 2, 3))[:, None] + \
                  entropy_penalty + usage_penalty * self.cfg.filter_usage_penalty + early_stop_penalty + runtime_penalty

        # print('states, new_states:', states.shape, new_states.shape)
        # print('penalty:', penalty.shape)

        if high_res is None:
            return (x, new_states, surrogate, penalty), debug_info, debugger
        else:
            return (x, new_states, high_res_output), debug_info, debugger


if __name__ == "__main__":
    # from easydict import EasyDict
    # cfg = EasyDict({"fc1_size": 4096, "curve_steps": 8})
    # ft = FeatureExtractor((14, 64, 64), 32, 4096, 0.5)
    # x = torch.randn((1, 14, 64, 64))
    # x = ft(x)
    # print(x.shape)
    from config import cfg
    print(cfg.curve_steps)
    batch = 1
    agent = Agent(cfg, (64, 64), 'cpu', meta_ccm=cfg.meta_ccm)
    x = torch.randn((batch, 3, 512, 512))
    z = torch.randn((batch, cfg.z_dim))
    states = torch.randn((batch, cfg.num_state_dim))
    agent((x, z, states), 0.1)
    print(agent.state_dict())
    # torch.save(agent.state_dict(), "agent.pth")
