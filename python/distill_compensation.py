"""独立 Compensator 蒸馏、恢复和配对评价，不保存渲染图片。"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from compensation_student import (StudentCompensator, decode_student, distillation_loss,
    load_student, save_student, teacher_fingerprint)


TRAIN_RATIOS = (.01, .05, .1, .3, .5, .7)
UNSEEN_RATIOS = (.03, .08, .2, .4, .6)


def ratio_controls(ratio, knots, bounds):
    weights = [1.] + [-1.] * (len(knots)-1)
    for j in range(1, len(knots)):
        if ratio <= knots[j-1] + 1e-8:
            break
        weights[j] = min(1., (ratio-knots[j-1])/(knots[j]-knots[j-1]))
    t = 2*(math.log(ratio)-math.log(bounds[0]))/(math.log(bounds[1])-math.log(bounds[0]))-1
    return weights, [t, math.sin(math.pi*t/2), math.cos(math.pi*t/2)]


class TeacherSamples:
    def __init__(self, checkpoint, dataset, seed=0):
        from budget_viewer import BudgetScene
        from deployment_runtime import DeploymentRenderer
        self.scene = BudgetScene(checkpoint, dataset, 1000, seed, None, 'pytorch')
        self.runtime = DeploymentRenderer(self.scene.model, self.scene.gaussians, self.scene.config, seed)
        self.fingerprint = teacher_fingerprint(self.runtime._base.model, self.runtime._base.pc)
        self.local_modes = [False, True] if self.runtime._base.joint_schema else [False]

    def camera(self, frame):
        from utils import MiniCam, focal2fov
        c = self.scene.calibration
        pose = self.scene.views[frame]['pose']
        return MiniCam(c['width'], c['height'], focal2fov(c['fx'], c['width']),
            focal2fov(c['fy'], c['height']), np.linalg.inv(np.asarray(pose)))

    @torch.no_grad()
    def sample(self, frame, ratio, local, max_points=0, generator=None):
        r = self.runtime
        b = r._base
        camera = self.camera(frame)
        if local:
            _, ids, _ = r.prepare(camera, ratio_budget=ratio, view_conditioned=True)
        else:
            # Preserve the project's global FlexGS threshold and tie semantics.
            _, ids, _ = r._reference.prepare(None, global_ratio=ratio, compensation=False, measure_time=False)
            r._reference.global_cache.clear()
        full_k = len(ids)
        if max_points and full_k > max_points:
            take = torch.randperm(full_k, generator=generator)[:max_points].to(ids.device)
            ids = ids[take.sort().values]
        if not len(ids):
            raise RuntimeError(f'Empty distillation selection at frame {frame}, ratio {ratio}, local {local}')
        view = b.empty
        if local and b.joint_schema != 1:
            view = r._ext.view(b.anchors.centers, b.view_centers4, b.view_corners4,
                camera.camera_center.contiguous(), camera.world_view_transform, camera.full_proj_transform,
                b.model.selector_view_scene_scale, camera.image_width/camera.image_height, camera.znear)
        weights, code = ratio_controls(ratio, r.knots, b.spatial.ratio_encoding_values)
        control = b.offset.new_tensor(weights + code)
        inputs = b.compensation_input(ids, control).detach()
        condition = None
        if local:
            values = view[b.owners[ids]] if b.joint_schema in (2, 4) else inputs.new_zeros((len(ids), 7))
            condition = torch.cat((values, values.new_ones((len(ids), 1))), dim=1)
        flat, target, status = r._ext.compensate(b.offset, ids, b.owners, b.reference, b.tables,
            r._compensator, b.compensated_master, view, r._view_weight, weights, code,
            local, b.joint_schema, hasattr(b.spatial, 'softplus_scale_schema'))
        if int(status) or not torch.isfinite(target).all():
            raise RuntimeError('Nonfinite frozen teacher targets')
        return dict(inputs=inputs, ratio_code=inputs.new_tensor([code]), reference=b.reference[ids].detach(),
            condition=condition, master=[v[ids].detach() for v in b.compensated_master],
            target=target.detach(), softplus=hasattr(b.spatial, 'softplus_scale_schema'),
            packet=r._packet(flat.detach(), len(ids)), ids=ids, frame=int(frame), ratio=float(ratio),
            local=bool(local), full_k=full_k, camera=camera)


def student_residual(student, case):
    return student(case['inputs'], case['ratio_code'], case['reference'], case['condition'])


def render_student(student, case, background, sh_degree, residual=None):
    from renderer import _make_rasterizer
    if residual is None:
        residual = student_residual(student, case)
    inputs = decode_student(residual, case['master'], case['softplus'])
    if any(not torch.isfinite(v).all() for v in inputs.values() if isinstance(v, torch.Tensor)):
        raise RuntimeError('Student produced nonfinite Renderer attributes')
    rasterizer = _make_rasterizer(case['camera'], sh_degree, background, 1., inference_only=False)
    return rasterizer(**inputs)[0]


def view_split(count):
    if count < 5:
        raise ValueError('At least five views are required for held-out validation')
    valid = list(range(0, count, 5))
    train = [i for i in range(count) if i % 5]
    return train, valid


def attribute_objective(losses, residual_weight=1.):
    return sum(value * (residual_weight if name == 'residual' else 1.)
        for name, value in losses.items())


def training_ratio(ratios, index, modes, seed, jitter=0.):
    pair = index // modes
    ratio = ratios[pair % len(ratios)]
    if jitter:
        generator = torch.Generator().manual_seed(seed + (pair+1)*104729)
        probability, interval, phase = torch.rand(3, generator=generator).tolist()
        if probability < jitter:
            interval = min(int(interval*(len(ratios)-1)), len(ratios)-2)
            low, high = ratios[interval:interval+2]
            ratio = math.exp(math.log(low) + phase*(math.log(high)-math.log(low)))
    return ratio


@torch.no_grad()
def validation_loss(student, source, views, ratios, seed, image_weight=0.,
                    residual_weight=1., metric='combined'):
    from renderer import render_prepared
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for frame in views:
        for ratio in ratios:
            for local in source.local_modes:
                render_image = bool(image_weight or metric == 'image_mse')
                case = source.sample(frame, ratio, local, 0 if render_image else 2048, generator)
                losses = distillation_loss(student, case, student_residual(student, case))
                if any(not torch.isfinite(v) for v in losses.values()):
                    raise RuntimeError('Nonfinite validation loss')
                score = attribute_objective(losses, residual_weight)
                if render_image:
                    teacher_image = render_prepared(case['camera'],case['packet'],source.scene.background)['render']
                    image = render_student(student,case,source.scene.background,source.runtime._base.pc.active_sh_degree)
                    image_mse = (image-teacher_image).square().mean()
                    score = image_mse if metric == 'image_mse' else score + image_weight*image_mse
                if not torch.isfinite(score):
                    raise RuntimeError('Nonfinite validation objective')
                rows.append(float(score))
    return float(np.mean(rows))


def train(args):
    output = args.output.resolve()
    if output.exists() and args.resume is None:
        raise ValueError('Use a new output directory or an explicit --resume checkpoint')
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    source = TeacherSamples(args.checkpoint, args.dataset, args.seed)
    train_views, valid_views = view_split(len(source.scene.views))
    valid_views = [valid_views[int(i)] for i in np.linspace(0, len(valid_views)-1, min(3, len(valid_views)), dtype=int)]
    config = dict(seed=args.seed, ratios=args.ratios, learning_rate=args.lr, batch_points=args.batch_points,
        image_every=args.image_every, image_weight=args.image_weight, calibration_cases=args.calibration_cases,
        validation_views=valid_views, validation_ratios=list(args.validation_ratios or args.ratios),
        architecture='51+ratio3+view8 -> 32 -> 32 -> four direct Heads',
        teacher_fingerprint=source.fingerprint, teacher_checkpoint=str(source.scene.checkpoint),
        frozen=['Master', 'Selector', 'Anchor Feature', 'normalization', 'topology'],
        target_backend='deployment_fp32_v31_teacher', teacher_binding='tensor_content_sha256')
    config['validation_image_weight'] = args.image_weight/args.image_every if args.image_every else 0.
    config['residual_weight'] = args.residual_weight
    config['validation_metric'] = args.validation_metric
    config['ratio_jitter'] = args.ratio_jitter
    config['paired_view_sampling'] = args.paired_view_sampling
    config['initial_student'] = str(args.initialize) if args.initialize is not None else None
    first_step = 0
    best = float('inf')
    if args.resume is not None:
        student, previous = load_student(args.resume, 'cuda', source.fingerprint)
        saved = previous['metadata']['configuration']
        for key in ['seed', 'ratios', 'learning_rate', 'batch_points', 'image_every', 'image_weight','validation_image_weight']:
            if key not in saved:
                raise ValueError('Resume validation objective changed. Start a new run with --initialize instead')
            if saved[key] != config[key]:
                raise ValueError('Resume configuration changed at ' + key)
        for key, default in [('residual_weight', 1.), ('validation_metric', 'combined'),
                ('ratio_jitter', 0.), ('paired_view_sampling', False), ('validation_ratios', list(args.ratios))]:
            if saved.get(key, default) != config[key]:
                raise ValueError('Resume configuration changed at ' + key)
        first_step = previous['step']
        config['initial_student'] = saved.get('initial_student')
        if previous.get('generator_state') is None or previous.get('optimizer') is None:
            raise ValueError('Resume checkpoint has no optimizer or sampling RNG state')
        generator.set_state(previous['generator_state'].cpu())
    elif args.initialize is not None:
        student, _ = load_student(args.initialize, 'cuda', source.fingerprint)
    else:
        student = StudentCompensator.from_teacher(source.runtime._base.spatial)
        student.teacher_fingerprint = source.fingerprint
        calibration = []
        for i in range(args.calibration_cases):
            frame = train_views[int(torch.randint(len(train_views), (), generator=generator))]
            ratio = args.ratios[(i // len(source.local_modes)) % len(args.ratios)]
            local = source.local_modes[i % len(source.local_modes)]
            calibration.append(source.sample(frame, ratio, local, min(args.batch_points, 1024), generator))
        student.calibrate(calibration)
        del calibration
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    if args.resume is not None:
        optimizer.load_state_dict(previous['optimizer'])
    if first_step >= args.steps:
        raise ValueError('Requested total steps must exceed the resume step')
    log_path = output/'training.csv'
    if args.resume is not None and log_path.exists():
        with log_path.open(newline='') as existing:
            steps = [int(row['step']) for row in csv.DictReader(existing)]
        if steps and max(steps) > first_step:
            raise ValueError('Output contains steps newer than this checkpoint. Resume into a new output directory')
    (output/'configuration.json').write_text(json.dumps(config, indent=2, ensure_ascii=False))
    start_validation = validation_loss(student, source, valid_views, config['validation_ratios'], args.seed ^ 761,
        config['validation_image_weight'], args.residual_weight, args.validation_metric)
    best = start_validation
    if args.resume is not None and (output/'best.pt').exists():
        _, prior_best = load_student(output/'best.pt', 'cpu', source.fingerprint)
        best = min(best, prior_best['metadata']['best_validation'])
    metadata = dict(configuration=config, best_validation=best,
        initial_validation=previous['metadata']['initial_validation'] if args.resume is not None else start_validation,
        resume_checkpoint=str(args.resume) if args.resume is not None else None, resume_step=first_step)
    if args.resume is None or not (output/'best.pt').exists():
        save_student(output/'best.pt', student, metadata=metadata, optimizer=optimizer,
            step=0, generator_state=generator.get_state())
    from renderer import render_prepared
    with log_path.open('a' if args.resume else 'w', newline='', encoding='utf-8') as log:
        writer = csv.DictWriter(log, fieldnames=['step','frame','ratio','local','K','full_K','total',
            'residual','xyz','scale','rotation','opacity','dc','image','image_ratio','image_local','validation'])
        if log.tell() == 0:
            writer.writeheader()
        for step in range(first_step, args.steps):
            frame_generator = (torch.Generator().manual_seed(args.seed+(step+1)*19349663)
                if args.paired_view_sampling else generator)
            frame = train_views[int(torch.randint(len(train_views), (), generator=frame_generator))]
            local = source.local_modes[step % len(source.local_modes)]
            ratio = training_ratio(args.ratios, step, len(source.local_modes), args.seed, args.ratio_jitter)
            image_step = bool(args.image_every and (step+1) % args.image_every == 0)
            case = source.sample(frame, ratio, local, args.batch_points, generator)
            optimizer.zero_grad(set_to_none=True)
            residual = student_residual(student, case)
            losses = distillation_loss(student, case, residual)
            loss = attribute_objective(losses, args.residual_weight)
            image_loss = loss.new_zeros(())
            image_ratio = image_local = None
            if image_step:
                image_index = (step+1)//args.image_every-1
                image_ratio = training_ratio(args.ratios, image_index, len(source.local_modes), args.seed, args.ratio_jitter)
                image_local = source.local_modes[image_index % len(source.local_modes)]
                image_case = source.sample(frame, image_ratio, image_local)
                with torch.no_grad():
                    teacher_image = render_prepared(image_case['camera'], image_case['packet'], source.scene.background)['render']
                image = render_student(student, image_case, source.scene.background, source.runtime._base.pc.active_sh_degree)
                image_loss = (image-teacher_image).square().mean()
                loss = loss + args.image_weight*image_loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite distillation loss at step ' + str(step+1))
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in student.parameters()):
                raise RuntimeError('Nonfinite student gradients at step ' + str(step+1))
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in student.parameters()):
                raise RuntimeError('Nonfinite student parameters at step ' + str(step+1))
            value = None
            if (step+1) % args.validate_every == 0 or step+1 == args.steps:
                value = validation_loss(student, source, valid_views, config['validation_ratios'], args.seed ^ 761,
                    config['validation_image_weight'], args.residual_weight, args.validation_metric)
                improved = value < best
                best = min(best, value)
                metadata.update(best_validation=best, latest_validation=value)
                if improved:
                    save_student(output/'best.pt', student, metadata=metadata, optimizer=optimizer,
                        step=step+1, generator_state=generator.get_state())
                save_student(output/'last.pt', student, metadata=metadata, optimizer=optimizer,
                    step=step+1, generator_state=generator.get_state())
                print(json.dumps(dict(step=step+1, loss=float(loss), validation=value, best=best)), flush=True)
            writer.writerow(dict(step=step+1, frame=frame, ratio=ratio, local=local, K=len(case['ids']),
                full_K=case['full_k'], total=float(loss), image=float(image_loss), validation=value,
                image_ratio=image_ratio, image_local=image_local,
                **{key: float(v) for key, v in losses.items()}))
            log.flush()
    if teacher_fingerprint(source.runtime._base.model, source.runtime._base.pc) != source.fingerprint:
        raise RuntimeError('Frozen teacher was changed during distillation')
    for tensor in list(source.runtime._base.model.parameters()) + source.runtime._base.base_master:
        if tensor.grad is not None:
            raise RuntimeError('Distillation leaked gradients into the teacher')
    print('DISTILLATION_COMPLETE', output, flush=True)


@torch.no_grad()
def evaluate(args):
    from deployment_runtime import DeploymentRenderer
    from renderer import render_prepared
    source = TeacherSamples(args.checkpoint, args.dataset, args.seed)
    student, payload = load_student(args.student, 'cuda', source.fingerprint)
    runtime = DeploymentRenderer(source.scene.model, source.scene.gaussians, source.scene.config, args.seed, student=student)
    _, views = view_split(len(source.scene.views))
    excluded = payload.get('metadata',{}).get('configuration',{}).get('validation_views',[])
    views = [frame for frame in views if frame not in excluded]
    if not views:
        raise ValueError('No independent evaluation views remain after excluding checkpoint-selection views')
    if args.frames:
        views = [views[int(i)] for i in np.linspace(0, len(views)-1, min(args.frames, len(views)), dtype=int)]
    if args.output.exists():
        raise ValueError('Evaluation output already exists')
    args.output.mkdir(parents=True)
    rows = []
    for frame in views:
        for ratio in args.ratios:
            for local in source.local_modes:
                case = source.sample(frame, ratio, local)
                target = render_prepared(case['camera'], case['packet'], source.scene.background)['render']
                if local:
                    packet, ids, _ = runtime.prepare(case['camera'], ratio_budget=ratio)
                    if not torch.equal(ids, case['ids']):
                        raise RuntimeError('Student deployment changed Selector identities')
                    image = render_prepared(case['camera'], packet, source.scene.background)['render']
                else:
                    image = render_student(student, case, source.scene.background, runtime._base.pc.active_sh_degree)
                residual = student_residual(student, case)
                metrics = distillation_loss(student, case, residual)
                gt = source.scene.ground_truth(frame)
                def psnr(a, b):
                    return -10*math.log10(max(float((a.clamp(0,1)-b.clamp(0,1)).square().mean()),1e-12))
                teacher_psnr = psnr(target, gt) if gt is not None else None
                student_psnr = psnr(image, gt) if gt is not None else None
                rows.append(dict(frame=frame, ratio=ratio, local=local, K=len(case['ids']),
                    teacher_student_psnr=psnr(image, target), teacher_gt_psnr=teacher_psnr,
                    student_gt_psnr=student_psnr, psnr_loss=None if gt is None else teacher_psnr-student_psnr,
                    **{key:float(value) for key,value in metrics.items()}))
    with (args.output/'quality.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries = []
    for ratio in args.ratios:
        for local in source.local_modes:
            group = [r for r in rows if r['ratio']==ratio and r['local']==local]
            losses = [r['psnr_loss'] for r in group if r['psnr_loss'] is not None]
            summaries.append(dict(ratio=ratio, local=local, views=len(group),
                teacher_student_psnr=float(np.mean([r['teacher_student_psnr'] for r in group])),
                mean_gt_psnr_loss=float(np.mean(losses)) if losses else None,
                p99_gt_psnr_loss=float(np.percentile(losses,99)) if losses else None,
                worst_gt_psnr_loss=max(losses) if losses else None,
                quality_gate_passed=bool(np.mean(losses)<=.01 and np.percentile(losses,99)<=.05 and max(losses)<=.1) if losses else None))
    report = dict(teacher_fingerprint=source.fingerprint, student_step=payload['step'], images_saved=0,
        speed_measured=False, frames=views, excluded_validation_views=excluded, summaries=summaries,
        quality_gate=dict(mean_psnr_loss=.01,p99_psnr_loss=.05,worst_psnr_loss=.1),
        all_quality_gates_passed=all(bool(r['quality_gate_passed']) for r in summaries))
    (args.output/'quality.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description='冻结教师的 Compensator32 独立蒸馏')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['train','evaluate']:
        p = commands.add_parser(name)
        p.add_argument('checkpoint', type=Path)
        p.add_argument('--dataset', type=Path)
        p.add_argument('--output', required=True, type=Path)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--ratios', nargs='+', type=float,
            default=list(TRAIN_RATIOS if name=='train' else TRAIN_RATIOS+UNSEEN_RATIOS))
    p = commands.choices['train']
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-points', type=int, default=4096)
    p.add_argument('--calibration-cases', type=int, default=12)
    p.add_argument('--image-every', type=int, default=5)
    p.add_argument('--image-weight', type=float, default=1000.)
    p.add_argument('--residual-weight', type=float, default=1., help='原始14D残差监督权重，解码后属性项保持原权重')
    p.add_argument('--validation-metric', choices=['combined', 'image_mse'], default='combined')
    p.add_argument('--validation-ratios', nargs='+', type=float)
    p.add_argument('--ratio-jitter', type=float, default=0., help='区间内对数均匀 Ratio 替换概率，0为原固定轮转')
    p.add_argument('--paired-view-sampling', action='store_true', help='视角由seed和步号独立生成，隔离不同K对采样RNG的影响')
    p.add_argument('--validate-every', type=int, default=500)
    restart = p.add_mutually_exclusive_group()
    restart.add_argument('--resume', type=Path)
    restart.add_argument('--initialize', type=Path, help='仅载入学生权重，新建优化器和训练调度')
    p = commands.choices['evaluate']
    p.add_argument('--student', required=True, type=Path)
    p.add_argument('--frames', type=int, default=0)
    args = parser.parse_args()
    if any(not math.isfinite(r) or not 0<r<=1 for r in args.ratios) or len(set(args.ratios)) != len(args.ratios):
        parser.error('ratios must be distinct values in (0,1]')
    if args.command == 'train' and args.validation_ratios is not None and (
            any(not math.isfinite(r) or not 0<r<=1 for r in args.validation_ratios) or
            len(set(args.validation_ratios)) != len(args.validation_ratios)):
        parser.error('validation ratios must be distinct values in (0,1]')
    if args.command=='train' and (min(args.steps,args.batch_points,args.calibration_cases,args.validate_every)<=0 or
            args.image_every<0 or not math.isfinite(args.lr) or args.lr<=0 or
            not math.isfinite(args.image_weight) or args.image_weight<0 or
            not math.isfinite(args.residual_weight) or args.residual_weight<0 or
            not math.isfinite(args.ratio_jitter) or not 0<=args.ratio_jitter<=1):
        parser.error('Invalid distillation budget or loss parameters')
    if args.command == 'train' and args.ratio_jitter and (
            len(args.ratios)<2 or args.ratios != sorted(args.ratios)):
        parser.error('ratio jitter requires at least two increasing ratios')
    if args.command=='evaluate' and args.frames<0:
        parser.error('frames must be nonnegative')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command=='train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
