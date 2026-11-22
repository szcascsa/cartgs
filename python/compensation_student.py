"""冻结 Residual48 教师的独立 32/32 蒸馏学生与属性监督。"""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


HEADS = (('xyz', 3), ('scale', 3), ('rotation', 4), ('other', 4))
SCHEMA = 'residual48_student32_direct_heads_v1'


def teacher_fingerprint(model, gaussians):
    """只在绑定 checkpoint 时执行，覆盖行序、Feature、Selector 与 Master。"""
    digest = hashlib.sha256()
    tensors = dict(model.state_dict())
    for name in ('_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest'):
        tensors['master.' + name] = getattr(gaussians, name)
    for name, value in sorted(tensors.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    spatial = model.spatial_compensation
    digest.update(json.dumps([list(model.anchors.compensation_values),
        list(spatial.ratio_encoding_values), gaussians.active_sh_degree,
        int(getattr(model, 'joint_view_conditioning_schema', 0)),
        hasattr(spatial, 'softplus_scale_schema'), hasattr(spatial, 'physical_scale_additive_schema')]).encode())
    return digest.hexdigest()


class StudentCompensator(nn.Module):
    """两层共享 32D，四个独立线性 Head，输入与输出沿用教师语义。"""
    def __init__(self, view_conditioned=True):
        super().__init__()
        self.encode = nn.Linear(51, 32)
        self.body = nn.Linear(32, 32)
        self.ratio_projection = nn.Linear(3, 32, bias=False)
        if view_conditioned:
            self.view_projection = nn.Linear(8, 32, bias=False)
        self.heads = nn.ModuleDict({name: nn.Linear(32, width) for name, width in HEADS})
        self.register_buffer('target_scale', torch.ones(14))
        self.teacher_fingerprint = None
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def hidden(self, inputs, ratio_code, condition=None):
        if (inputs.ndim != 2 or inputs.shape[1] != 51 or ratio_code.ndim != 2 or
                ratio_code.shape[1] != 3 or ratio_code.shape[0] not in (1, len(inputs))):
            raise ValueError('Student requires input Kx51 and ratio 1x3 or Kx3')
        value = self.encode(inputs.detach()) + self.ratio_projection(ratio_code.detach())
        if condition is not None:
            if not hasattr(self, 'view_projection') or condition.shape != (len(inputs), 8):
                raise ValueError('Student view condition must be Kx8 and match its schema')
            value = value + self.view_projection(condition.detach())
        return F.relu(self.body(F.relu(value)))

    def unscaled(self, inputs, ratio_code, condition=None):
        hidden = self.hidden(inputs, ratio_code, condition)
        return torch.cat([head(hidden) for head in self.heads.values()], dim=1)

    def forward(self, inputs, ratio_code, reference, condition=None):
        if reference.shape != (len(inputs), 1):
            raise ValueError('Student reference must be Kx1')
        value = self.unscaled(inputs, ratio_code, condition)
        return torch.cat((value[:, :3] * reference.detach(), value[:, 3:]), dim=1)

    @classmethod
    def from_teacher(cls, spatial):
        if (spatial.encode.in_features != 51 or spatial.encode.out_features != 64 or
                spatial.body.in_features != 64 or spatial.body.out_features != 64):
            raise ValueError('Distillation requires the Residual48 64/64 teacher')
        model = cls(hasattr(spatial, 'view_projection')).to(spatial.encode.weight)
        with torch.no_grad():
            model.encode.weight.copy_(spatial.encode.weight[:32])
            model.encode.bias.copy_(spatial.encode.bias[:32])
            model.body.weight.copy_(spatial.body.weight[:32, :32])
            model.body.bias.copy_(spatial.body.bias[:32])
            model.ratio_projection.weight.copy_(spatial.ratio_projection.weight[:32])
            if hasattr(model, 'view_projection'):
                model.view_projection.weight.copy_(spatial.view_projection.weight[:32])
        return model

    def deployment_parameters(self):
        values = [self.encode.weight, self.encode.bias, self.body.weight, self.body.bias,
                  self.ratio_projection.weight]
        for head in self.heads.values():
            values.extend((head.weight, head.bias))
        return values

    @torch.no_grad()
    def calibrate(self, cases, ridge=1e-4):
        """用训练样本拟合四个输出层，避免从零补偿开始的大偏差。"""
        features, targets = [], []
        for case in cases:
            features.append(self.hidden(case['inputs'], case['ratio_code'], case['condition']).cpu().double())
            y = case['target'].detach().clone()
            y[:, :3] /= case['reference']
            targets.append(y.cpu().double())
        x, y = torch.cat(features), torch.cat(targets)
        if not len(x):
            raise ValueError('Calibration requires nonempty training samples')
        x = torch.cat((x, torch.ones((len(x), 1), dtype=x.dtype)), dim=1)
        penalty = torch.eye(33, dtype=x.dtype) * (ridge * len(x))
        penalty[-1, -1] = 0
        coefficient = torch.linalg.solve(x.t() @ x + penalty, x.t() @ y)
        begin = 0
        for name, width in HEADS:
            self.heads[name].weight.copy_(coefficient[:32, begin:begin+width].t())
            self.heads[name].bias.copy_(coefficient[32, begin:begin+width])
            begin += width
        self.target_scale.copy_(y.square().mean(dim=0).sqrt().clamp_min(1e-3))


def decode_student(residual, master, softplus):
    from transform_field import _multiply_quaternion, _normalize_quaternion
    xyz, logits, scale, rotation, dc, rest = [v.detach() for v in master]
    identity = torch.zeros_like(residual[:, 6:10])
    identity[:, 0] = 1
    rotation = _normalize_quaternion(_multiply_quaternion(
        _normalize_quaternion(identity + residual[:, 6:10]), rotation))
    rotation = _normalize_quaternion(rotation)
    if softplus:
        scale = (scale * F.softplus(residual[:, 3:6] + 0.5413248546129181)).clamp_min(torch.finfo(scale.dtype).tiny)
    else:
        scale = scale + residual[:, 3:6]
    return dict(means3D=xyz + residual[:, :3], opacities=(logits + residual[:, 10:11]).sigmoid(),
        scales=scale.log().exp(), rotations=rotation, dc=dc + residual[:, 11:14].unsqueeze(1),
        shs=rest, means2D=torch.zeros_like(xyz), colors_precomp=None, cov3D_precomp=None)


def distillation_loss(student, case, residual):
    """教师、Master、Feature 与输入一律 detach，仅学生可反传。"""
    reference = case['reference'].detach()
    target = case['target'].detach()
    difference = torch.cat(((residual[:, :3] - target[:, :3]) / reference,
                            residual[:, 3:] - target[:, 3:]), dim=1)
    normalized = difference / student.target_scale.detach()
    losses = dict(residual=normalized.square().mean())
    pred = decode_student(residual, case['master'], case['softplus'])
    with torch.no_grad():
        truth = decode_student(target, case['master'], case['softplus'])
    losses['xyz'] = ((pred['means3D'] - truth['means3D']) / reference).square().mean()
    base_scale = case['master'][2].detach()
    if case['softplus']:
        physical = (base_scale * F.softplus(residual[:, 3:6] + 0.5413248546129181)).clamp_min(torch.finfo(base_scale.dtype).tiny)
    else:
        physical = base_scale + residual[:, 3:6]
    floor = torch.finfo(base_scale.dtype).tiny
    losses['scale'] = (physical.clamp_min(floor).log() - truth['scales'].log()).square().mean()
    losses['scale'] = losses['scale'] + (F.relu(floor-physical) / base_scale.clamp_min(1e-6)).square().mean()
    losses['rotation'] = (1 - (pred['rotations'] * truth['rotations']).sum(dim=1).square()).clamp_min(0).mean()
    losses['opacity'] = (pred['opacities'] - truth['opacities']).square().mean()
    losses['dc'] = (pred['dc'] - truth['dc']).square().mean()
    return losses


def save_student(path, student, *, metadata, optimizer=None, step=0, generator_state=None):
    if not student.teacher_fingerprint:
        raise ValueError('Bind the student to a frozen teacher before saving')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema=SCHEMA, view_conditioned=hasattr(student, 'view_projection'),
        teacher_fingerprint=student.teacher_fingerprint,
        state_dict={k: v.detach().cpu().clone() for k, v in student.state_dict().items()},
        metadata=metadata, step=int(step), optimizer=None if optimizer is None else optimizer.state_dict(),
        generator_state=generator_state)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_student(path, device='cpu', expected_fingerprint=None):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get('schema') != SCHEMA or not isinstance(payload.get('view_conditioned'), bool):
        raise ValueError('Unsupported Compensation student checkpoint')
    fingerprint = payload.get('teacher_fingerprint')
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError('Student checkpoint has no valid teacher binding')
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError('Student checkpoint belongs to a different teacher, map or Gaussian row order')
    student = StudentCompensator(payload['view_conditioned']).to(device)
    student.load_state_dict(payload['state_dict'], strict=True)
    if any(not torch.isfinite(v).all() for v in student.state_dict().values()) or not (student.target_scale > 0).all():
        raise ValueError('Nonfinite student state or invalid target normalization')
    student.teacher_fingerprint = fingerprint
    return student, payload
