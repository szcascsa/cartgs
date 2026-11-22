import os
import argparse
import numpy as np
import torch
from tqdm import trange
import json
import glob
from renderer import prepare_render_data, render_prepared
from argparse import ArgumentParser
from gaussian_model import GaussianModel
from selector import build_selector_mask, load_selector, load_selector_metadata, compensated_attributes
from transform_field import (
    load_transform_field,
    transformed_attributes,
)
from scipy.spatial.transform import Rotation
from utils import MiniCam, focal2fov

from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import cv2

from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from evo.tools import plot

import matplotlib.pyplot as plt
import copy
import csv
import time

calc_psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
calc_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
calc_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()


def loadReplica(path):
    color_paths = sorted(glob.glob(os.path.join(path, "results/frame*.jpg")))
    tstamp = [
        float(
            color_path.split("/")[-1]
            .replace("frame", "")
            .replace(".jpg", "")
            .replace(".png", "")
        )
        for color_path in color_paths
    ]
    return color_paths, tstamp


def loadTUM(path):
    # Use the index timestamps when image names are frame numbers.
    rgb_txt = os.path.join(path, "rgb.txt")
    if os.path.isfile(rgb_txt):
        color_paths = []
        tstamp = []
        with open(rgb_txt, "r", encoding="utf-8-sig") as fin:
            for line_number, line in enumerate(fin, 1):
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(
                        "Invalid RGB entry at {}:{}: expected timestamp and image path".format(
                            rgb_txt, line_number
                        )
                    )
                timestamp = float(parts[0])
                image_path = os.path.join(path, parts[1])
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(image_path)
                color_paths.append(image_path)
                tstamp.append(timestamp)
        if not color_paths:
            raise RuntimeError("No RGB images found in " + rgb_txt)
        print("[loadTUM] Loaded {} RGB images from {}".format(len(color_paths), rgb_txt))
        return color_paths, np.asarray(tstamp, dtype=np.float64)

    rgb_dir = os.path.join(path, "rgb3" if os.path.exists(os.path.join(path, "rgb3")) else "rgb")
    color_paths = sorted(
        color_path
        for color_path in glob.glob(os.path.join(rgb_dir, "*"))
        if os.path.splitext(color_path)[1].lower() in (".png", ".jpg", ".jpeg")
        and os.path.isfile(color_path)
    )
    if not color_paths:
        raise RuntimeError("No RGB images found in " + rgb_dir)
    tstamp = [
        float(os.path.splitext(os.path.basename(color_path))[0].replace("frame", ""))
        for color_path in color_paths
    ]
    return color_paths, tstamp


def loadKITTI(path):
    color_paths = sorted(glob.glob(os.path.join(path, "image_2/*.png")))
    tstamp = np.loadtxt(
        os.path.join(path, "times.txt"), delimiter=" ", dtype=np.str_
    ).astype(np.float32)
    return color_paths, tstamp


def loadEuRoC(path):
    color_paths = sorted(glob.glob(os.path.join(path, "mav0/cam0/data/*.png")))
    tstamp = [np.float64(x.split("/")[-1][:-4]) / 1e9 for x in color_paths]
    return color_paths, tstamp


def associate_frames(tstamp_image, tstamp_pose, max_dt=0.08):
    """Pair images, depths, and poses."""
    associations = []
    for i, t in enumerate(tstamp_image):
        j = np.argmin(np.abs(tstamp_pose - t))
        if np.abs(tstamp_pose[j] - t) < max_dt:
            associations.append((i, j))
    return associations


def load_selector_config(config_path):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            "selector experiment config not found: {}".format(config_path)
        )
    storage = cv2.FileStorage(config_path, cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError("failed to open selector experiment config: {}".format(config_path))

    def read_value(key, default):
        node = storage.getNode(key)
        if node.empty():
            return default
        return node.real() if isinstance(default, float) else int(node.real())

    config = {
        "backend": storage.getNode("Selector.backend").string() if not storage.getNode("Selector.backend").empty() else "cuda",
        "selection_mode": storage.getNode("Selector.selection_mode").string() if not storage.getNode("Selector.selection_mode").empty() else "cached_bernoulli",
        "gumbel_budget_normalize": read_value("Selector.gumbel_budget_normalize", 0),
        "anchor_frustum_culling": read_value("Selector.anchor_frustum_culling", 0),
        "drift_temperature": read_value("Protection.drift_temperature", 0.10),
        "beta_fast": read_value("Protection.beta_fast", 0.60),
        "beta_slow": read_value("Protection.beta_slow", 0.15),
        "compatibility_mature_threshold": read_value("Protection.compatibility_mature_threshold", 0.80),
        "protection_enabled": read_value("Selector.protection_enabled", 1),
        "temperature": read_value("Selector.temperature", 1.0),
        "entry_maturity": read_value("Elastic.entry_maturity", 0.0),
    }
    storage.release()
    if config['selection_mode'] not in ('cached_bernoulli', 'flexgs'):
        raise ValueError('Selector.selection_mode must be cached_bernoulli or flexgs')
    return config


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="evaluation script parameters")
    parser.add_argument("result_path", type=str, default=None)
    parser.add_argument("gt_path", type=str, default=None)
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="write metrics/images to this directory while reading the model from result_path",
    )
    parser.add_argument("--selector", type=str, default=None)
    parser.add_argument("--ply", type=str, default=None,
                        help="render a standalone PLY without Selector or Transform; use result_path for cameras/trajectory")
    parser.add_argument("--shutdown-model", default=None,
                        help="select a specific shutdown checkpoint for Full evaluation")
    parser.add_argument("--selector-ratio", type=float, default=None)
    parser.add_argument("--selector-config", type=str, default=None)
    parser.add_argument(
        "--selector-protection-enabled",
        type=int,
        choices=(0, 1),
        default=None,
    )
    parser.add_argument("--selector-temperature", type=float, default=None)
    parser.add_argument("--correct_scale", action="store_true")
    parser.add_argument("--stereo-camera-config", default=None,
                        help="evaluate Stereo against rectified left images using the training calibration")
    parser.add_argument("--stereo-native-maps", default=None,
                        help="optional FP32 map archive exported by the training OpenCV runtime")
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--evaluation-frame-set", default=None,
                        help="JSON gt_indices shared by paired ablations; never selects frames by quality")
    parser.add_argument("--no-save-images", action="store_true",
                        help="compute metrics without writing rendered or ground-truth images")
    parser.add_argument("--selector-anchor-frustum", action=argparse.BooleanOptionalAction, default=None,
                        help="opt in to per-view nominal Anchor AABB candidates; disabled by default")
    parser.add_argument("--selector-sampling-rng", default=None,
                        help="saved selector_sampling_rng.pt for matched fixed per-Gaussian Gumbel noise")
    parser.add_argument("--inference-budget", type=int, default=None,
                        help="opt-in strict point budget over visible nominal Anchor members; ratio=B/N")
    parser.add_argument("--budget-sampling-seed", type=int, default=None,
                        help="fixed Gaussian Gumbel keys for point-budget rendering; default 0")
    args = parser.parse_args()
    if args.stereo_native_maps and not args.stereo_camera_config:
        parser.error("--stereo-native-maps requires --stereo-camera-config")
    source_result_path = os.path.abspath(args.result_path)
    output_result_path = os.path.abspath(args.output_path or args.result_path)
    os.makedirs(output_result_path, exist_ok=True)
    if args.selector and args.selector_ratio is None and args.inference_budget is None:
        parser.error("--selector requires --selector-ratio")
    if args.budget_sampling_seed is not None and args.inference_budget is None:
        parser.error("--budget-sampling-seed requires --inference-budget")
    if args.inference_budget is not None:
        if not args.selector or args.inference_budget < 0:
            parser.error("--inference-budget requires --selector and a nonnegative integer")
        if args.selector_ratio is not None or args.selector_sampling_rng is not None:
            parser.error("point-budget mode computes ratio=B/N and uses --budget-sampling-seed")
        if args.selector_anchor_frustum is False:
            parser.error("point-budget mode requires nominal Anchor frustum culling")
        args.selector_anchor_frustum = True
    if args.ply and args.selector:
        parser.error("--ply cannot be combined with --selector; exported PLYs are already compensated")
    if args.selector_anchor_frustum and not args.selector:
        parser.error("--selector-anchor-frustum requires --selector")
    sh_degree = 3
    gaussians = GaussianModel(sh_degree)
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    selector_model = (
        load_selector(args.selector, "cuda") if args.selector else None
    )
    shutdown_dirs = [
        name for name in os.listdir(source_result_path)
        if name.endswith("_shutdown") and name.split("_", 1)[0].isdigit()
    ]
    if not shutdown_dirs:
        raise FileNotFoundError("no shutdown model found in {}".format(source_result_path))
    if selector_model is not None:
        checkpoint_dir = os.path.normcase(
            os.path.abspath(os.path.dirname(args.selector))
        )
        matches = [
            name for name in shutdown_dirs
            if checkpoint_dir
            in {
                os.path.normcase(
                    os.path.abspath(os.path.join(source_result_path, name))
                ),
                os.path.normcase(
                    os.path.abspath(os.path.join(source_result_path, name, "ply"))
                ),
            }
        ]
        if len(matches) != 1:
            raise ValueError(
                "selector.pt must belong to a shutdown model or its ply directory "
                "in result_path"
            )
        shutdown_dir = matches[0]
    elif args.shutdown_model:
        if args.shutdown_model not in shutdown_dirs:
            raise ValueError("shutdown model not found: {}".format(args.shutdown_model))
        shutdown_dir = args.shutdown_model
    else:
        shutdown_dir = max(shutdown_dirs, key=lambda name: int(name.split("_", 1)[0]))
    shutdown_ply_dir = os.path.join(source_result_path, shutdown_dir, "ply")
    selector_config_path = args.selector_config or os.path.join(
        shutdown_ply_dir, "selector_config.yaml"
    )
    selector_config = load_selector_config(selector_config_path) if selector_model is not None else {}
    if args.selector_anchor_frustum is None:
        args.selector_anchor_frustum = False
    if args.selector_sampling_rng and not args.selector_anchor_frustum:
        parser.error("--selector-sampling-rng requires Anchor frustum selection")
    if selector_model is not None:
        selector_model.set_backend(selector_config["backend"])
    selector_protection_enabled = (
        args.selector_protection_enabled
        if args.selector_protection_enabled is not None
        else selector_config.get("protection_enabled", True)
    )
    selector_temperature = (
        args.selector_temperature
        if args.selector_temperature is not None
        else selector_config.get("temperature", 1.0)
    )
    transform_model = (
        load_transform_field(shutdown_ply_dir, "cuda")
        if selector_model is not None else None
    )
    # load model
    width, height, fovx, fovy = 0, 0, 0, 0
    ts = []
    Rs = []
    for file_name in (shutdown_dir,):
        if "shutdown" in file_name:
            iter = file_name.split("_")[0]
            ply_path = os.path.join(
                source_result_path,
                file_name,
                "ply/point_cloud/iteration_{}".format(iter),
                "point_cloud.ply",
            )
            render_ply_path = os.path.abspath(args.ply or ply_path)
            gaussians.load_ply(render_ply_path)
            with open(
                os.path.join(
                    source_result_path, file_name, "ply", "cameras.json"
                ),
                "r",
            ) as fin:
                camera_paras = json.load(fin)

            width, height, fx, fy = (
                camera_paras[0]["width"],
                camera_paras[0]["height"],
                camera_paras[0]["fx"],
                camera_paras[0]["fy"],
            )
            fovx = focal2fov(fx, width)
            fovy = focal2fov(fy, height)

            selector_mask = None
            selector_stats = None
            if selector_model is None:
                total = int(gaussians.get_xyz.shape[0])
                selector_stats = {"active_gaussians": total, "total_gaussians": total,
                                  "target_total_gaussians": total, "actual_vs_target": 1.0,
                                  "transform_applied": False}
            elastic_attributes = None
            if selector_model is not None and not args.selector_anchor_frustum:
                metadata = None
                metadata_path = os.path.join(
                    source_result_path,
                    file_name,
                    "ply",
                    "point_cloud",
                    "iteration_{}".format(iter),
                    "selector_metadata.ply",
                )
                metadata = load_selector_metadata(
                    metadata_path, "cuda"
                )
                shared=None
                if hasattr(selector_model,'compensation') or hasattr(selector_model,'spatial_compensation'):
                    if transform_model is not None:
                        raise RuntimeError('Selector compensation and Transform Field are mutually exclusive')
                    with torch.no_grad():
                        shared=selector_model.forward_selection_features(gaussians.get_xyz,gaussians.get_scaling,
                            gaussians.get_features[:,:1],gaussians.get_opacity,
                            torch.tensor([[args.selector_ratio]],device='cuda'),selector_temperature,
                            return_logits=selector_config['selection_mode']=='flexgs')
                extra=dict(forward_features=shared) if shared is not None else {}
                if selector_config['selection_mode']=='flexgs':
                    torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()),
                               os.path.join(output_result_path,'selector_sampling_rng.pt'))
                selector_mask, selector_stats = build_selector_mask(
                    selector_model, gaussians, metadata, args.selector_ratio, selector_config,
                    selector_temperature, protection_enabled=bool(selector_protection_enabled),**extra)
                if shared is not None:
                    elastic_attributes=compensated_attributes(selector_model,gaussians,selector_mask,args.selector_ratio,shared[1])
                    selector_stats['selector_compensation_applied']=elastic_attributes is not None
                    selector_stats['compensation_decoded_rows']=int(selector_mask.sum())
                    selector_stats['compensation_output_dim']=14 if hasattr(selector_model,'spatial_compensation') else selector_model.compensation[-1].out_features
                    selector_stats['appearance_compensation_applied']=elastic_attributes is not None and 'opacity_logits' in elastic_attributes
                if selector_config['selection_mode']=='flexgs':
                    np.save(os.path.join(output_result_path,'selector_selected_ids.npy'),
                            torch.nonzero(selector_mask>0).flatten().cpu().numpy())
                if transform_model is not None:
                    maturity = metadata["maturity"].detach() if selector_protection_enabled else torch.ones_like(metadata["maturity"])
                    elastic_attributes = transformed_attributes(
                        transform_model, gaussians, selector_mask, maturity, args.selector_ratio)
                    selector_stats["transform_applied"] = elastic_attributes is not None

                selector_stats.setdefault("transform_applied", False)
                render_ply_path = os.path.join(output_result_path, "point_cloud.ply")
                exported_count = gaussians.save_ply(
                    render_ply_path, selector_mask=selector_mask, elastic_attributes=elastic_attributes
                )
                manifest = {
                    "ply_file": "point_cloud.ply",
                    "ratio": args.selector_ratio,
                    "source_ply": os.path.abspath(ply_path),
                    "selector_checkpoint": os.path.abspath(args.selector),
                    "selector_config": os.path.abspath(selector_config_path),
                    "selector_temperature": selector_temperature,
                    "selector_selection_mode": selector_config['selection_mode'],
                    "selector_sampling_seed": selector_stats.get('sampling_seed'),
                    "protection_enabled": bool(selector_protection_enabled),
                    "transform_checkpoint": os.path.join(shutdown_ply_dir, "transform_field.pt")
                        if transform_model is not None else None,
                    "active_gaussians": exported_count,
                    "total_gaussians": selector_stats["total_gaussians"],
                    "transform_applied": transform_model is not None and elastic_attributes is not None,
                    "selector_compensation_applied": shared is not None and elastic_attributes is not None,
                    "appearance_compensation_applied": elastic_attributes is not None and 'opacity_logits' in elastic_attributes,
                }
                with open(os.path.join(output_result_path, "ply_manifest.json"), "w") as fout:
                    json.dump(manifest, fout, indent=2)
                gaussians = GaussianModel(sh_degree)
                gaussians.load_ply(render_ply_path)
                if gaussians.get_xyz.shape[0] != exported_count:
                    raise RuntimeError("reloaded PLY count does not match exported selection")
                selector_mask = None
                elastic_attributes = None
                selector_model = None
                transform_model = None
                metadata = None

    stereo_evaluation = None
    if args.stereo_camera_config:
        from stereo_evaluation import StereoEvaluation
        stereo_evaluation = StereoEvaluation(args.stereo_camera_config,width,height,fx,fy,args.stereo_native_maps)
    with open(os.path.join(output_result_path,"image_preprocessing.json"),"w",encoding="utf-8") as stream:
        json.dump(stereo_evaluation.metadata if stereo_evaluation else {"protocol":"original_rgb"},stream,indent=2)
    frustum_renderer = None
    if args.inference_budget is not None:
        from anchor_budget import FixedBudgetRenderer
        if transform_model is not None:
            raise ValueError('Point-budget mode supports Selector compensation only')
        budget_config = dict(selector_config, temperature=selector_temperature,
                             protection_enabled=bool(selector_protection_enabled))
        frustum_renderer = FixedBudgetRenderer(selector_model, gaussians, budget_config,
            0 if args.budget_sampling_seed is None else args.budget_sampling_seed)
        selector_stats = dict(total_gaussians=len(gaussians.get_xyz), ratio=None,
            fixed_point_budget=True, requested_budget=args.inference_budget,
            per_view_selection=True, selection_mode='flexgs', selection_policy='visible_ratio_B_over_N_stable_topk',
            selector_compensation_applied=True, rasterizer_mode='inference',
            ply_path=render_ply_path, nominal_bounds='center +/- offsetScale/2; no padding')
        render_data = None
    elif args.selector_anchor_frustum:
        from anchor_frustum import AnchorFrustumRenderer
        if transform_model is not None:
            raise ValueError('Anchor-frustum mode currently supports Selector compensation only')
        if args.selector_sampling_rng:
            state = torch.load(args.selector_sampling_rng, map_location='cpu', weights_only=False)
            torch.set_rng_state(state['cpu'])
            torch.cuda.set_rng_state(state['cuda'])
        torch.save(dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state()),
                   os.path.join(output_result_path, 'selector_sampling_rng.pt'))
        frustum_renderer = AnchorFrustumRenderer(selector_model, gaussians, args.selector_ratio,
            selector_config, selector_temperature, bool(selector_protection_enabled))
        selector_stats = dict(total_gaussians=len(gaussians.get_xyz), ratio=float(args.selector_ratio),
            target_active=float(args.selector_ratio)*len(gaussians.get_xyz), budget_population=len(gaussians.get_xyz),
            selection_mode='flexgs', selection_policy='anchor_nominal_frustum_global_quantile',
            per_view_selection=True, nominal_bounds='center +/- offsetScale/2; no padding',
            frustum='camera full projection six planes, conservative AABB test',
            selector_compensation_applied=True, rasterizer_mode='inference',
            sampling_rng_source=args.selector_sampling_rng, ply_path=render_ply_path)
        render_data = None
    else:
        selector_stats.update(ply_path=render_ply_path,
                              render_gaussians=int(gaussians.get_xyz.shape[0]), rasterizer_mode="inference")
        render_data = prepare_render_data(gaussians)

    # load gt
    if "replica" in args.gt_path.lower():
        gt_color_paths, gt_tstamp = loadReplica(args.gt_path)
    elif "kitti" in args.gt_path.lower():
        gt_color_paths, gt_tstamp = loadKITTI(args.gt_path)
    elif "euroc" in args.gt_path.lower():
        print(args.gt_path)
        gt_color_paths, gt_tstamp = loadEuRoC(args.gt_path)
    else:
        gt_color_paths, gt_tstamp = loadTUM(args.gt_path)

    #### pose evaluation
    # load estimated poses
    pose_path = os.path.join(source_result_path, "CameraTrajectory_TUM.txt")
    traj_est = file_interface.read_tum_trajectory_file(pose_path)
    # load gt pose
    if "kitti" in args.gt_path.lower():

        def loadKITTIPose(gt_path):
            scene = gt_path.split("/")[-1]
            gt_file = gt_path.replace(scene, "poses/{}.txt".format(scene))
            pose_quat = []
            with open(gt_file, "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    line = lines[i].split()
                    # print(line)
                    c2w = np.array(list(map(float, line))).reshape(3, 4)
                    # print(c2w)
                    quat = np.zeros(7)
                    quat[:3] = c2w[:3, 3]
                    quat[3:] = Rotation.from_matrix(c2w[:3, :3]).as_quat()
                    pose_quat.append(quat)
            pose_quat = np.array(pose_quat)

            return pose_quat

        pose_quat = loadKITTIPose(args.gt_path)
        traj_ref = PoseTrajectory3D(
            positions_xyz=pose_quat[:, :3],
            orientations_quat_wxyz=pose_quat[:, 3:],
            timestamps=np.array(gt_tstamp),
        )

    elif "replica" in args.gt_path.lower():
        gt_file = os.path.join(args.gt_path, "pose_TUM.txt")
        traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    elif "euroc" in args.gt_path.lower():
        gt_file = os.path.join(
            args.gt_path, "mav0/state_groundtruth_estimate0/data.csv"
        )
        traj_ref = file_interface.read_euroc_csv_trajectory(gt_file)
        T_i_c0 = np.array(
            [
                [
                    0.0148655429818,
                    -0.999880929698,
                    0.00414029679422,
                    -0.0216401454975,
                ],
                [
                    0.999557249008,
                    0.0149672133247,
                    0.025715529948,
                    -0.064676986768,
                ],
                [
                    -0.0257744366974,
                    0.00375618835797,
                    0.999660727178,
                    0.00981073058949,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        traj_ref.transform(T_i_c0, True)
    else:
        gt_file = os.path.join(args.gt_path, "groundtruth.txt")
        traj_ref = file_interface.read_tum_trajectory_file(gt_file)

    traj_ref, traj_est = sync.associate_trajectories(
        traj_ref, traj_est, max_diff=0.08
    )
    traj_ref.align(traj_est, True)
    poses = traj_est.poses_se3
    result = main_ape.ape(
        traj_ref,
        traj_est,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=args.correct_scale,
    )
    result_rotation_part = main_ape.ape(
        traj_ref,
        traj_est,
        est_name="rot",
        pose_relation=PoseRelation.rotation_part,
        align=True,
        correct_scale=args.correct_scale,
    )

    out_path = os.path.join(output_result_path, "metrics_traj.txt")
    with open(out_path, "w") as fp:
        fp.write(result.pretty_str())
        fp.write(result_rotation_part.pretty_str())
    print(result)

    if args.show_plot:
        traj_est_aligned = copy.deepcopy(traj_est)
        traj_est_aligned.align(traj_ref, correct_scale=True)
        fig = plt.figure()
        traj_by_label = {
            "estimate (not aligned)": traj_est,
            "estimate (aligned)": traj_est_aligned,
            "reference": traj_ref,
        }
        plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
        plt.show()

    ## render and evaluation
    tstamp = traj_est.timestamps
    associations = associate_frames(tstamp, gt_tstamp)
    if args.evaluation_frame_set:
        with open(args.evaluation_frame_set, encoding="utf-8") as stream:
            requested = set(json.load(stream)["gt_indices"])
        associations = [pair for pair in associations if pair[1] in requested]
        if len(associations) != len(requested) or {pair[1] for pair in associations} != requested:
            raise ValueError("run does not cover the exact shared evaluation frame set")
    if not associations:
        raise ValueError("no associated frames available for rendering evaluation")
    with open(os.path.join(output_result_path, "evaluation_frames.json"), "w", encoding="utf-8") as stream:
        json.dump({"gt_indices": [int(pair[1]) for pair in associations],
                   "rgb_paths": [gt_color_paths[pair[1]] for pair in associations]}, stream, indent=2)

    if not args.no_save_images:
        os.makedirs(os.path.join(output_result_path, "image"), exist_ok=True)
    if not args.no_save_images and "_0" in output_result_path:
        os.makedirs(os.path.join(output_result_path, "gt"), exist_ok=True)
    psnr_list, ssim_list, lpips_list, time_list = [], [], [], []
    frustum_stats, selected_rows = [], []
    for index in trange(
        len(associations),
        desc="rendering {}".format(output_result_path.split("/")[-1]),
    ):
        (result_indx, gt_indx) = associations[index]
        w2c = np.linalg.inv(poses[result_indx])
        cam = MiniCam(width, height, fovx, fovy, w2c)
        if frustum_renderer is not None:
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            prepare_start = time.perf_counter()
            start_event.record()
            if args.inference_budget is not None:
                render_data, selected_ids, frame_stats = frustum_renderer.prepare(cam, args.inference_budget)
            else:
                render_data, selected_ids, frame_stats = frustum_renderer.prepare(cam)
            end_event.record()
            prepare_wall_ms = (time.perf_counter()-prepare_start)*1000
        if index == 0:
            for _ in range(10):
                render_prepared(cam, render_data, background)
        render_result = render_prepared(cam, render_data, background, measure_time=True,
                                       return_radii=args.inference_budget is not None)
        if frustum_renderer is not None:
            frame_stats = dict(trajectory_index=int(result_indx),gt_index=int(gt_indx),
                **frame_stats, selector_prepare_cuda_ms=start_event.elapsed_time(end_event),
                selector_prepare_wall_ms=prepare_wall_ms, rendering_ms=render_result['rendering_ms'])
            if args.inference_budget is not None:
                frame_stats['prepare_plus_render_ms'] = prepare_wall_ms + render_result['rendering_ms']
                frame_stats['projected_gaussians'] = int((render_result['radii'] > 0).sum())
            frustum_stats.append(frame_stats)
            selected_rows.append(selected_ids.cpu().numpy().astype(np.int32,copy=False))
        render_image = render_result["render"]
        render_image = render_image.permute(1, 2, 0)
        gt_image = cv2.imread(gt_color_paths[gt_indx], cv2.IMREAD_COLOR)
        if stereo_evaluation is not None:
            gt_image = stereo_evaluation.transform(gt_image)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        if len(gt_image.shape) < 3:
            gt_image = np.broadcast_to(
                gt_image[..., None], (gt_image.shape[0], gt_image.shape[1], 3)
            )

        render_image = torch.clamp(render_image, 0.0, 1.0)
        if not args.no_save_images:
            predict_image_np = render_image.detach().cpu().numpy()
            predict_image_img = np.uint8(predict_image_np * 255)
            predict_image_img = cv2.cvtColor(predict_image_img, cv2.COLOR_BGR2RGB)

        gt_image_torch = (
            torch.from_numpy(np.array(gt_image))
            .to("cuda")
            .permute(2, 0, 1)[None]
            / 255.0
        )
        render_image_torch = render_image.permute([2, 0, 1])[None]
        val_psnr = calc_psnr(render_image_torch, gt_image_torch).item()
        val_ssim = calc_ssim(render_image_torch, gt_image_torch).item()
        val_lpips = calc_lpips(render_image_torch, gt_image_torch).item()

        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        if not args.no_save_images and "_0" in output_result_path:
            cv2.imwrite(
                os.path.join(
                    output_result_path,
                    "gt",
                    gt_color_paths[gt_indx].split("/")[-1],
                ),
                gt_image,
            )
        if not args.no_save_images:
            cv2.imwrite(
                os.path.join(
                    output_result_path,
                    "image",
                    gt_color_paths[gt_indx].split("/")[-1],
                ),
                predict_image_img,
            )

        psnr_list.append(val_psnr)
        ssim_list.append(val_ssim)
        lpips_list.append(val_lpips)
        time_list.append(render_result["rendering_ms"])

    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    time_list = np.array(time_list)
    np.savetxt(os.path.join(output_result_path, "psnr.txt"), psnr_list)
    np.savetxt(os.path.join(output_result_path, "ssim.txt"), ssim_list)
    np.savetxt(os.path.join(output_result_path, "lpips.txt"), lpips_list)
    np.savetxt(
        os.path.join(output_result_path, "render_time.txt"),
        np.column_stack(([pair[0] for pair in associations], time_list)),
        fmt=("%d", "%.8f"),
        header="trajectory frame index, render time(milliseconds); synchronized rasterizer, 10 warmup renders excluded",
    )

    with open(os.path.join(source_result_path, "TrackingTime.txt"), "r") as fin:
        tracking_time = fin.readlines()
    tracking_time = np.array(tracking_time[:-3]).astype(np.float32)

    with open(os.path.join(output_result_path, "eval.txt"), "w") as fout:
        fout.write("psnr: {}\n".format(np.mean(psnr_list)))
        fout.write("ssim: {}\n".format(np.mean(ssim_list)))
        fout.write("lpips: {}\n".format(np.mean(lpips_list)))
        fout.write("tracking s: {}\n".format(np.mean(tracking_time)))
        fout.write("tracking FPS: {}\n".format(1 / np.mean(tracking_time)))

        fout.write("rendering ms: {}\n".format(np.mean(time_list)))
        fout.write("rendering FPS: {}\n".format(1000 / np.mean(time_list)))

    if frustum_renderer is not None:
        per_view_file = 'anchor_budget_per_view.csv' if args.inference_budget is not None else 'anchor_frustum_per_view.csv'
        with open(os.path.join(output_result_path,per_view_file),'w',newline='') as stream:
            writer=csv.DictWriter(stream,list(frustum_stats[0]));writer.writeheader();writer.writerows(frustum_stats)
        offsets=np.concatenate(([0],np.cumsum([len(x) for x in selected_rows],dtype=np.int64)))
        np.savez_compressed(os.path.join(output_result_path,'selector_selected_ids_by_view.npz'),
            offsets=offsets,ids=np.concatenate(selected_rows),gt_indices=np.array([p[1] for p in associations]))
        aggregate_keys = ['visible_anchors','candidate_gaussians','eligible_candidates','active_gaussians',
                          'budget_shortfall','selector_prepare_cuda_ms','selector_prepare_wall_ms']
        if args.inference_budget is not None:
            aggregate_keys += ['effective_budget', 'projected_gaussians', 'selector_ms', 'compensation_ms',
                               'prepare_plus_render_ms']
        for key in aggregate_keys:
            values=np.array([x[key] for x in frustum_stats],dtype=np.float64)
            selector_stats.update({key+'_mean':float(values.mean()),key+'_min':float(values.min()),
                                   key+'_max':float(values.max())})
        if args.inference_budget is not None:
            ratios = [x['ratio'] for x in frustum_stats if x['ratio'] is not None]
            selector_stats.update(active_gaussians=selector_stats['active_gaussians_mean'],
                actual_active=selector_stats['active_gaussians_mean'],
                render_gaussians=selector_stats['active_gaussians_mean'], views=len(frustum_stats),
                ratio_mean=float(np.mean(ratios)) if ratios else None,
                actual_vs_target=selector_stats['active_gaussians_mean']/max(1, args.inference_budget),
                outside_training_range_views=sum(not x['ratio_in_training_range'] for x in frustum_stats))
        else:
            selector_stats.update(active_gaussians=selector_stats['active_gaussians_mean'],
                actual_active=selector_stats['active_gaussians_mean'],
                render_gaussians=selector_stats['active_gaussians_mean'],
                actual_vs_target=selector_stats['active_gaussians_mean']/max(1.,selector_stats['target_active']),
                views=len(frustum_stats),budget_shortfall_views=sum(x['budget_shortfall']>0 for x in frustum_stats),
                legacy_quantile_budget=frustum_stats[0]['legacy_quantile_budget'])
    if selector_stats is not None:
        with open(
            os.path.join(output_result_path, "selector_stats.json"), "w"
        ) as fout:
            json.dump(selector_stats, fout, indent=2)
