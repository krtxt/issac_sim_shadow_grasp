"""
单次Isaac Gym稳定性仿真脚本 (核心版本)

职责：从task_spec读取配置、准备资产、运行仿真、保存结果、清理资源
"""
import os
import sys
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import json
import argparse
import traceback
import gc
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

from isaacgym import gymapi, gymutil, gymtorch
from envs.tasks.grasp_test_force_shadowhand import IsaacGraspTestForce_shadowhand as IsaacGraspTestForce
import torch
import numpy as np
import cv2
import trimesh as tm
from plotly import graph_objects as go
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from utils.handmodel import get_handmodel
from utils.rot6d import rot_to_orthod6d

DEFAULT_TEMP_ROOT = Path('explore_issac/temp')

STABILITY_CONFIG = {
    'task': {'useStage': False, 'useSlider': False, 'useGroundTruth': False},
    'env': {
        'env_name': 'grasp', 'numTrain': 6, 'numObservations': 50, 'numActions': 9,
        'envSpacing': 1.0, 'maxEpisodeLength': 256, 'actionSpeedScale': 20,
        'enableDebugVis': False, 'contactBufferSize': 9999, 'contactMovingThreshold': 0.1,
        'pointFeatureDim': 512, 'objPointDownsampleNum': 2048, 'handPointDownsampleNum': 64,
        'pointNetLR': 0.0001, 'visualizePointcloud': False, 'enableCameraSensors': False,
        'depth_bar': 10, 'map_dis_bar': 0.1, 'moving_pc_mode': False, 'driveMode': 'pos',
        'clipObservations': 5.0, 'clipActions': 1.0,
        'asset': {'assetRoot': 'envs/assets', 'AssetNumTrain': 1,
                  'trainObjAssets': {0: {'name': 0, 'path': 'object_to_grasp/apple/apple.urdf'}}},
    },
    'sim': {
        'substeps': 2,
        'physx': {'num_threads': 4, 'solver_type': 1, 'num_position_iterations': 8,
                  'num_velocity_iterations': 0, 'contact_offset': 0.002, 'rest_offset': 0.0,
                  'bounce_threshold_velocity': 0.2, 'max_depenetration_velocity': 1000.0,
                  'default_buffer_size_multiplier': 5.0},
        'flex': {'num_outer_iterations': 5, 'num_inner_iterations': 20, 'warm_start': 0.8, 'relaxation': 0.75},
    },
    'object': {'density': 1000, 'damping': {'linear': 10, 'angular': 100}, 'shape': {'friction': 2.}},
    'agent': {'density': 1000, 'dof_props': {'stiffness': 400.0, 'velocity': 0.8, 'damping': 400.0},
              'shape': {'friction': 2.}},
    'eval_policy': {
        'init': {'steps': 200},
        'dynamic': {'directions': {0: [1., 0., 0.], 1: [-1., 0., 0.], 2: [0., 1., 0.],
                                    3: [0., -1., 0.], 4: [0., 0., 1.], 5: [0., 0., -1.]},
                    'num_steps': 50, 'magnitude_per_volume': 500.},
        'error': {'distance': 0.02},
    },
}

console = Console()


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='单次Isaac Gym稳定性仿真')
    parser.add_argument('--task_spec', type=str, required=True, help='任务配置文件路径(JSON)')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--record_dir', type=str, default=None, help='录制目录（默认使用output_dir）')
    parser.add_argument('--onscreen', action='store_true', default=False, help='显示可视化窗口')
    parser.add_argument('--cpu', action='store_true', default=False, help='使用CPU运行')
    parser.add_argument('--debug', action='store_true', default=False, help='调试模式')
    parser.add_argument('--static_preview', action='store_true', default=False,
                        help='仅加载静态手-物体姿态并录制预览视频，不执行稳定性仿真')
    return parser.parse_args()


def get_sim_param():
    """创建Isaac Gym仿真参数"""
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    sim_params.num_client_threads = 0
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True
    sim_params.physx.num_subscenes = 0
    sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    sim_params.physx.num_threads = 0
    return sim_params


def prepare_scaled_asset(source_dir: str, scale: float, temp_root: Path) -> Path:
    """拷贝物体资产到临时目录并缩放URDF"""
    source_dir = Path(source_dir)
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"{source_dir.parent.name}_{uuid.uuid4().hex}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    shutil.copytree(source_dir, temp_dir)

    urdf_path = temp_dir / 'coacd.urdf'
    if urdf_path.exists():
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for mesh in root.findall('.//mesh'):
            scale_attr = mesh.get('scale', '1 1 1')
            parts = scale_attr.replace(',', ' ').split()
            values = []
            for part in parts:
                try:
                    values.append(float(part))
                except ValueError:
                    values.append(1.0)
            if len(values) != 3:
                values = [1.0, 1.0, 1.0]
            scaled_values = [val * scale for val in values]
            mesh.set('scale', ' '.join(f"{val:.8f}" for val in scaled_values))
        tree.write(urdf_path)

    return temp_dir.resolve()


def cleanup_all_isaac_resources(device='cuda', aggressive=False):
    """清理所有Isaac Gym资源"""
    if device != 'cpu' and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.reset_accumulated_memory_stats(device)
        except Exception as e:
            console.print(f"[bold yellow][WARNING] CUDA cleanup error:[/] {e}")
    
    for _ in range(3 if aggressive else 1):
        gc.collect()
    
    time.sleep(0.3 if aggressive else 0.1)


def cleanup_isaac_env(env, device='cuda', delay=0.2):
    """清理Isaac环境引用和CUDA缓存"""
    if env is None:
        return

    tensor_attrs = [
        'root_tensor', 'dof_state_tensor', 'rigid_body_tensor',
        'object_root_tensor', 'dexterous_root_tensor', 'dexterous_dof_tensor',
        'initial_dof_states', 'initial_root_states',
        'pos_act', 'eff_act', 'pos_action', 'eff_action',
        'recorded_frames', 'cameras', 'camera_tensors'
    ]

    for attr in tensor_attrs:
        if hasattr(env, attr):
            try:
                obj = getattr(env, attr)
                if torch.is_tensor(obj):
                    del obj
                elif isinstance(obj, list):
                    obj.clear()
                setattr(env, attr, None)
            except Exception:
                pass

    if hasattr(env, 'env_ptr_list'):
        try:
            env.env_ptr_list.clear()
        except Exception:
            pass

    try:
        del env
    except Exception:
        pass

    if device != 'cpu' and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        except Exception:
            pass

    gc.collect()
    if delay > 0:
        time.sleep(delay)


def build_opt_q(qpos_batch, device):
    """构建优化后的qpos张量"""
    q = torch.tensor(np.stack(qpos_batch), dtype=torch.float32)
    if q.shape[1] == 27:
        t = q[:, :3]
        a = q[:, 3:]
        rot6d = torch.tensor([[1., 0., 0., 0., 1., 0.]], dtype=torch.float32).repeat(q.shape[0], 1)
        opt_q = torch.cat([t, rot6d, a], dim=1)
    elif q.shape[1] == 25:
        t = q[:, :3]
        a22 = q[:, 3:]
        pad = torch.zeros((q.shape[0], 2), dtype=torch.float32)
        a = torch.cat([pad, a22], dim=1)
        rot6d = torch.tensor([[1., 0., 0., 0., 1., 0.]], dtype=torch.float32).repeat(q.shape[0], 1)
        opt_q = torch.cat([t, rot6d, a], dim=1)
    elif q.shape[1] >= 31:
        t = q[:, :3]
        rot6d = q[:, 3:9]
        if q.shape[1] == 31:
            a22 = q[:, 9:]
            pad = torch.zeros((q.shape[0], 2), dtype=torch.float32)
            a = torch.cat([pad, a22], dim=1)
        else:
            a = q[:, 9:]
        opt_q = torch.cat([t, rot6d, a], dim=1)
    else:
        raise ValueError(f'Unsupported qpos dimension: {q.shape[1]}')

    target_device = torch.device(device)
    opt_q = opt_q.to(target_device)
    return opt_q


def load_task_spec(task_spec_path):
    """加载任务配置文件"""
    with open(task_spec_path, 'r') as f:
        task_spec = json.load(f)
    
    required_fields = ['task_id', 'object_name', 'object_root', 'qpos_batch', 'scale_list']
    for field in required_fields:
        if field not in task_spec:
            raise ValueError(f"task_spec缺少必需字段: {field}")
    
    return task_spec


def save_results(output_dir, task_spec, results, timestamp):
    """保存仿真结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存metrics.json
    metrics = {
        'task_id': task_spec['task_id'],
        'object_name': task_spec['object_name'],
        'timestamp': timestamp,
        'total': results['total'],
        'succ_all': results['succ_all'],
        'succ_any': results['succ_any'],
        'rate_all': results['rate_all'],
        'rate_any': results['rate_any'],
    }
    
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    console.print(f"[green]结果已保存:[/] {metrics_path}")
    return metrics_path


def save_visualization(output_dir, task_spec, qpos_batch, mesh_path, scale, timestamp, device):
    """保存Plotly可视化HTML"""
    try:
        vis_dir = os.path.join(output_dir, timestamp)
        os.makedirs(vis_dir, exist_ok=True)
        
        html_filename = f"{timestamp}_{task_spec['object_name']}.html"
        html_path = os.path.join(vis_dir, html_filename)
        
        # 构造HandModel并用第一条grasp姿态可视化
        hand_model = get_handmodel(batch_size=1, device=device)
        opt_q = build_opt_q([qpos_batch[0]], device)
        hand_traces = hand_model.get_plotly_data(q=opt_q, i=0, color='lightblue', opacity=0.8)
        
        # 加载物体mesh
        mesh_for_vis = tm.load(mesh_path)
        mesh_for_vis.apply_scale(scale)
        obj_verts = mesh_for_vis.vertices
        obj_faces = mesh_for_vis.faces
        obj_trace = go.Mesh3d(
            x=obj_verts[:, 0], y=obj_verts[:, 1], z=obj_verts[:, 2],
            i=obj_faces[:, 0], j=obj_faces[:, 1], k=obj_faces[:, 2],
            color='orange', opacity=0.5,
        )
        
        fig = go.Figure(data=[obj_trace] + hand_traces)
        fig.write_html(str(html_path))
        console.print(f"[green]已保存Plotly HTML:[/] {html_path}")
        return html_path
    except Exception as exc:
        console.print(f"[bold red]Plotly可视化失败:[/] {exc}")
        if task_spec.get('debug', False):
            console.print(traceback.format_exc(), style="dim")
        return None


def render_result_summary(results):
    """使用 Rich 呈现稳定性测试结果概览"""
    table = Table(title="稳定性测试结果", box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("指标", justify="left")
    table.add_column("值", justify="right")
    table.add_row("抓取总数", str(results['total']))
    table.add_row("全部成功数", str(results['succ_all']))
    table.add_row("任一成功数", str(results['succ_any']))
    table.add_row("全部成功率", f"{results['rate_all']*100:.2f}%")
    table.add_row("任一成功率", f"{results['rate_any']*100:.2f}%")
    console.print(table)


class EvalIsaacEnv(IsaacGraspTestForce):
    """Isaac 环境封装，支持录制视频、手部单独录制和增强的资源清理。

    注意：该实现来自原 test_sta_new.py 中的 _EvalIsaac 类，做成独立可复用版本。
    """

    def __init__(
        self,
        *args,
        enable_recording: bool = False,
        record_dir: str = None,
        object_scale: float = 1.0,
        object_asset_dir: str = None,
        debug: bool = False,
        **kwargs,
    ):
        self.enable_recording = enable_recording
        self.record_dir = record_dir
        self.cameras = []
        self.camera_tensors = []
        self.recorded_frames = []
        self.object_scale = object_scale
        self.object_asset_dir = object_asset_dir
        self._cam_use_tensors = False
        self._cam_size = (1280, 720)
        self.debug = debug
        super().__init__(*args, **kwargs)

        if self.enable_recording:
            self._setup_cameras()

    def _load_obj_asset(self):
        """加载物体资产（覆盖父类中硬编码 Realdex 路径的实现）。

        优先使用 DexGraspNet 的 meshdata/<object_name>/coacd 目录中的
        decomposed.obj / coacd.urdf（通过 object_asset_dir 传入），如果不存在，
        再回退到 cfg['object_root']/<object_name>.obj/.urdf。
        """
        self.obj_name_list = []
        self.obj_asset_list = []
        self.table_asset_list = []
        self.obj_pose_list = []
        self.table_pose_list = []
        self.obj_actor_list = []

        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = self.cfg["object"]["density"]
        object_asset_options.linear_damping = self.cfg["object"]["damping"]["linear"]
        object_asset_options.angular_damping = self.cfg["object"]["damping"]["angular"]
        object_asset_options.fix_base_link = self.fix_object
        object_asset_options.disable_gravity = True
        object_asset_options.use_mesh_materials = True
        object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        object_asset_options.override_com = True
        object_asset_options.override_inertia = True
        object_asset_options.vhacd_enabled = False
        object_asset_options.vhacd_params = gymapi.VhacdParams()
        object_asset_options.vhacd_params.resolution = 1000000

        mesh_path = None
        urdf_path = None

        # 优先使用 coacd 目录（通过 object_asset_dir 传入）
        if self.object_asset_dir is not None:
            cand_obj = os.path.join(self.object_asset_dir, "decomposed.obj")
            cand_urdf = os.path.join(self.object_asset_dir, "coacd.urdf")
            if os.path.exists(cand_obj) and os.path.exists(cand_urdf):
                mesh_path = cand_obj
                urdf_path = cand_urdf

        # 否则退回到 object_root / <object_name>.obj/.urdf
        if mesh_path is None or urdf_path is None:
            root = self.cfg.get("object_root", ".")
            cand_obj = os.path.join(root, f"{self.object_name}.obj")
            cand_urdf = os.path.join(root, f"{self.object_name}.urdf")
            if not (os.path.exists(cand_obj) and os.path.exists(cand_urdf)):
                raise FileNotFoundError(
                    f"未找到物体资源. 尝试的路径: {cand_urdf} 与 {cand_obj} 或 coacd 目录 {self.object_asset_dir}"
                )
            mesh_path = cand_obj
            urdf_path = cand_urdf

        # 最终路径存在性检查
        if not os.path.exists(urdf_path) or not os.path.exists(mesh_path):
            raise FileNotFoundError(f"物体URDF或网格不存在: {urdf_path} | {mesh_path}")

        if self.debug:
            console.print(f"[cyan][DEBUG][load_obj_asset] object_name={self.object_name}[/]")
            console.print(f"[cyan][DEBUG][load_obj_asset] use_coacd_dir={self.object_asset_dir is not None}[/]")
            console.print(f"[cyan][DEBUG][load_obj_asset] urdf_path={urdf_path}[/]")
            console.print(f"[cyan][DEBUG][load_obj_asset] mesh_path={mesh_path}[/]")

        obj_asset = self.gym.load_asset(self.sim, "", urdf_path, object_asset_options)
        if obj_asset is None:
            raise RuntimeError(f"Isaac Gym 无法加载 URDF: {urdf_path}")

        self.object_mesh = tm.load(mesh_path)
        if hasattr(self, "object_scale") and self.object_scale is not None:
            self.object_mesh.apply_scale(self.object_scale)
        self.obj_asset_list.append(obj_asset)

        rig_dict = self.gym.get_asset_rigid_body_dict(obj_asset)
        if not rig_dict:
            raise RuntimeError(f"物体资源没有刚体: {urdf_path}")
        if self.debug:
            console.print(f"[cyan][DEBUG][load_obj_asset] rigid_bodies={list(rig_dict.keys())}[/]")

        self.obj_rig_name = list(rig_dict.keys())[0]
        obj_start_pose = gymapi.Transform()
        obj_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        obj_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.obj_pose_list.append(obj_start_pose)

    def _load_obj(self, env_ptr, env_id):
        """覆盖父类的 _load_obj，使用本地加载的资产列表，并去掉对刚体名为 'object' 的断言。"""
        if self.obj_loaded is False:
            self._load_obj_asset()
            self.obj_loaded = True

        obj_type = env_id // self.env_per_object
        subenv_id = env_id % self.env_per_object
        obj_actor = self.gym.create_actor(
            env_ptr,
            self.obj_asset_list[obj_type],
            self.obj_pose_list[obj_type],
            f"obj{obj_type}-{subenv_id}",
            env_id,
            0,
            0,
        )

        obj_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, obj_actor)
        for shape in obj_shape_props:
            shape.friction = self.cfg["object"]["shape"]["friction"]
        self.gym.set_actor_rigid_shape_properties(env_ptr, obj_actor, obj_shape_props)

        self.obj_actor_list.append(obj_actor)

    def _setup_cameras(self):
        """设置相机用于录制视频"""
        if len(self.env_ptr_list) > 0:
            env_ptr = self.env_ptr_list[0]

            camera_props = gymapi.CameraProperties()
            camera_props.width = 1280
            camera_props.height = 720
            # 为降低 GPU pipeline + camera tensor 的不稳定性，这里不启用图像张量访问
            camera_props.enable_tensors = False
            self._cam_use_tensors = False
            self._cam_size = (camera_props.width, camera_props.height)

            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)

            cam_pos = gymapi.Vec3(0.5, -0.5, 0.3)
            cam_target = gymapi.Vec3(0, 0, 0.1)
            self.gym.set_camera_location(camera_handle, env_ptr, cam_pos, cam_target)

            self.cameras.append(camera_handle)
            if self.debug:
                console.print(f"[cyan]视频录制相机已设置: 分辨率 {camera_props.width}x{camera_props.height}[/]")

    def capture_frame(self):
        """捕获当前帧"""
        if not self.enable_recording or len(self.cameras) == 0:
            return

        self.gym.render_all_camera_sensors(self.sim)
        started = False
        if self._cam_use_tensors:
            try:
                self.gym.start_access_image_tensors(self.sim)
                started = True
            except Exception:
                started = False
        try:
            for _cam_idx, camera_handle in enumerate(self.cameras):
                env_ptr = self.env_ptr_list[0]
                cam_img = self.gym.get_camera_image(self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
                h, w = self._cam_size[1], self._cam_size[0]
                try:
                    img_array = cam_img.reshape(h, w, 4)
                except Exception:
                    img_array = cam_img
                img_bgr = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2BGR)
                self.recorded_frames.append(img_bgr.copy())
        finally:
            if self._cam_use_tensors and started:
                try:
                    self.gym.end_access_image_tensors(self.sim)
                except Exception:
                    pass

    def save_video(self, output_path, fps: int = 30):
        """将录制的帧保存为视频文件"""
        if len(self.recorded_frames) == 0:
            console.print("[bold yellow]警告: 没有录制到任何帧[/]")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        height, width = self.recorded_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        console.print(Panel.fit(f"输出: {output_path}", title="正在保存视频", border_style="cyan"))
        console.print(f"[dim]帧数: {len(self.recorded_frames)} | 分辨率: {width}x{height} | FPS: {fps}[/]")

        for frame in self.recorded_frames:
            video_writer.write(frame)

        video_writer.release()
        console.print(f"[green]视频已保存:[/] {output_path}")
        self.recorded_frames.clear()

    def hide_objects(self):
        """隐藏所有物体，只保留手部（使用tensor API，兼容GPU pipeline）"""
        if not hasattr(self, "root_tensor") or self.root_tensor is None:
            return

        try:
            self._saved_object_root_states = self.root_tensor[:, 1, :].clone()
        except Exception:
            self._saved_object_root_states = None

        try:
            self.root_tensor[:, 1, 0] = 0.0
            self.root_tensor[:, 1, 1] = 0.0
            self.root_tensor[:, 1, 2] = -100.0
            self.root_tensor[:, 1, 7:10] = 0.0
            self.root_tensor[:, 1, 10:13] = 0.0
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))
        except Exception:
            pass

    def show_objects(self):
        """显示物体（恢复到隐藏前的根状态，如果有保存的话）"""
        if not hasattr(self, "root_tensor") or self.root_tensor is None:
            return

        if getattr(self, "_saved_object_root_states", None) is None:
            return

        try:
            self.root_tensor[:, 1, :] = self._saved_object_root_states
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))
        except Exception:
            pass

    def _apply_static_pose(self, static_pose):
        """将静态姿态写入所有环境的DoF，不触发仿真"""
        if static_pose is None:
            return False
        try:
            repeated_pose = static_pose.to(self.device).repeat(self.num_envs, 1)
            pose_dim = repeated_pose.shape[1]
            self.dof_state_tensor[:, :pose_dim, 0] = repeated_pose
            self.dof_state_tensor[:, :pose_dim, 1] = 0.0
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state_tensor))
            self.gym.refresh_dof_state_tensor(self.sim)
            return True
        except Exception:
            return False

    def record_hand_only(self, output_path, num_frames: int = 60, fps: int = 30):
        """录制只有手部的视频"""
        if not self.enable_recording:
            console.print("[bold yellow]警告: 录制功能未启用[/]")
            return

        console.print("\n[bold cyan]开始录制只有手部的视频...[/]")

        original_frames = self.recorded_frames.copy()
        self.recorded_frames.clear()

        self.hide_objects()

        static_pose = getattr(self, "static_hand_pose", None)
        static_mode = self._apply_static_pose(static_pose)

        for _i in range(num_frames):
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            if not static_mode:
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)

            if not self.headless:
                self.render()

            if self.cfg["env"].get("enableCameraSensors"):
                self.gym.step_graphics(self.sim)

            self.capture_frame()

            if getattr(self, "viewer", None) is not None:
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)

        if len(self.recorded_frames) > 0:
            self.save_video(output_path, fps=fps)
            console.print(f"[green]已保存只有手部的视频:[/] {output_path}")

        self.show_objects()
        self.recorded_frames = original_frames

    def record_static_scene(self, output_path, num_frames: int = 150, fps: int = 30,
                            include_object: bool = True):
        """录制带物体的静态手部预览视频"""
        if not self.enable_recording:
            console.print("[bold yellow]警告: 录制功能未启用，无法录制静态预览[/]")
            return

        console.print("\n[bold cyan]开始录制静态预览视频...[/]")
        original_frames = self.recorded_frames.copy()
        self.recorded_frames.clear()

        static_pose = getattr(self, "static_hand_pose", None)
        static_mode = self._apply_static_pose(static_pose)

        if not include_object:
            self.hide_objects()

        for _ in range(num_frames):
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            if not static_mode:
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)

            if not self.headless:
                self.render()
            if self.cfg["env"].get("enableCameraSensors"):
                self.gym.step_graphics(self.sim)

            self.capture_frame()

            if getattr(self, "viewer", None) is not None:
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)

        if len(self.recorded_frames) > 0:
            self.save_video(output_path, fps=fps)
            console.print(f"[green]已保存静态预览视频:[/] {output_path}")

        if not include_object:
            self.show_objects()
        self.recorded_frames = original_frames

    def close(self):
        """释放录制相关资源（相机与缓冲区）。

        仿真本身的销毁交由 Isaac/BaseTask 的析构逻辑处理，避免重复 destroy_sim/destroy_viewer
        带来的潜在 double-free 问题。
        """
        # 防御性结束图像张量访问（当前 _cam_use_tensors 为 False，一般不会进入）
        if self._cam_use_tensors and hasattr(self, "gym") and hasattr(self, "sim"):
            try:
                if self.gym is not None and self.sim is not None:
                    self.gym.end_access_image_tensors(self.sim)
            except Exception:
                pass

        # 显式销毁相机传感器
        if self.enable_recording and self.cameras:
            if hasattr(self, "gym") and self.gym is not None and hasattr(self, "env_ptr_list"):
                for env_ptr, cam_handle in zip(self.env_ptr_list, self.cameras):
                    try:
                        self.gym.destroy_camera_sensor(env_ptr, cam_handle)
                    except Exception:
                        pass
            self.cameras.clear()
            self.camera_tensors.clear()

        # 清理录制帧缓冲
        if hasattr(self, "recorded_frames") and isinstance(self.recorded_frames, list):
            self.recorded_frames.clear()

    def _push_object_with_direction(self, i_direction, pbar):
        """重写父类方法以支持视频录制"""
        object_force_magnitude = (
            self.cfg["eval_policy"]["dynamic"]["magnitude_per_volume"] * self.object_volume
        )
        object_pos_start = self.get_obj_pos()

        for _i_iter in range(self.force_num_steps):
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            force_position = self.rigid_body_tensor[:, :, :3].clone()
            device_loc = force_position.device
            object_force = torch.zeros_like(force_position, device=device_loc)
            direction_vec = torch.tensor(i_direction, device=device_loc)
            object_force[:, -1, :] = object_force_magnitude * direction_vec
            self.gym.apply_rigid_body_force_at_pos_tensors(
                self.sim,
                gymtorch.unwrap_tensor(object_force),
                gymtorch.unwrap_tensor(force_position),
                gymapi.ENV_SPACE,
            )
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            if not self.headless:
                self.render()

            if self.cfg["env"].get("enableCameraSensors") is True:
                self.gym.step_graphics(self.sim)

            if self.enable_recording:
                self.capture_frame()

            if getattr(self, "viewer", None) is not None:
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)
            pbar.update()

        # 与基类行为保持一致：根据物体位移判断是否稳定
        object_pos_terminal = self.get_obj_pos()
        return self.is_obj_stable(object_pos_start, object_pos_terminal)


def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建日志文件
    log_path = os.path.join(args.output_dir, 'log.txt')
    log_file = open(log_path, 'w')
    
    def log_print(msg, *, style=None, panel_title=None):
        """使用 Rich 同步输出到控制台和日志文件"""
        if panel_title:
            console.print(Panel.fit(msg, title=panel_title, border_style=style or "cyan"))
        elif style:
            console.print(f"[{style}]{msg}[/{style}]")
        else:
            console.print(msg)
        log_file.write(msg + '\n')
        log_file.flush()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_asset_dir = None
    
    try:
        log_print(f"========== 任务开始: {timestamp} ==========")
        log_print(f"task_spec: {args.task_spec}")
        log_print(f"output_dir: {args.output_dir}")
        
        # 加载任务配置
        task_spec = load_task_spec(args.task_spec)
        log_print(f"任务ID: {task_spec['task_id']}")
        log_print(f"物体名称: {task_spec['object_name']}")
        log_print(f"抓取数量: {len(task_spec['qpos_batch'])}")
        
        # 设备配置
        device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
        log_print(f"使用设备: {device}")
        
        # 准备物体资产
        object_name = task_spec['object_name']
        object_root = task_spec['object_root']
        scale = task_spec['scale_list'][0] if task_spec['scale_list'] else 1.0

        # 尝试使用 DexGraspNet 的 coacd 目录: object_root/<object_name>/coacd
        mesh_dir = os.path.join(object_root, object_name, 'coacd')
        has_coacd = os.path.isdir(mesh_dir) and os.path.exists(os.path.join(mesh_dir, 'coacd.urdf'))

        if has_coacd:
            mesh_path = os.path.join(mesh_dir, 'decomposed.obj')
            urdf_path = os.path.join(mesh_dir, 'coacd.urdf')
            if not (os.path.exists(mesh_path) and os.path.exists(urdf_path)):
                has_coacd = False

        if not has_coacd:
            mesh_path = os.path.join(object_root, f'{object_name}.obj')
            urdf_path = os.path.join(object_root, f'{object_name}.urdf')
            if not (os.path.exists(mesh_path) and os.path.exists(urdf_path)):
                raise FileNotFoundError(f"缺少物体资源: {mesh_path} 或 {urdf_path}")

        # 计算物体体积（使用选定 mesh 并按 scale 缩放）
        try:
            mesh = tm.load(mesh_path)
            mesh.apply_scale(scale)
            object_volume = mesh.volume
            log_print(f"物体体积: {object_volume:.6f}")
        except Exception:
            object_volume = 0.0
            log_print("警告: 无法计算物体体积，使用默认值0")

        # 若存在coacd资产，仿照 test_sta_new 生成缩放后的临时目录供仿真使用
        asset_dir_to_use = None
        if has_coacd:
            try:
                temp_asset_dir = prepare_scaled_asset(mesh_dir, scale, DEFAULT_TEMP_ROOT)
                asset_dir_to_use = str(temp_asset_dir)
                if args.debug:
                    log_print(f"[DEBUG] 使用缩放资产: src={mesh_dir}, scale={scale:.6f}, temp={asset_dir_to_use}")
            except Exception as exc:
                asset_dir_to_use = mesh_dir
                log_print(f"警告: 生成缩放资产失败，改用未缩放资产 ({object_name}): {exc}")
        
        # 构建qpos
        qpos_batch = task_spec['qpos_batch']
        opt_q = build_opt_q(qpos_batch, device)
        log_print(f"构建qpos完成: shape={opt_q.shape}")
        
        # 配置Isaac环境
        cfg = dict(STABILITY_CONFIG)
        cfg['object_root'] = object_root
        
        # 启用相机用于录制
        enable_recording = task_spec.get('record_options', {}).get('enable_recording', False)
        if enable_recording:
            cfg['env']['enableCameraSensors'] = True
        
        # 创建仿真环境
        log_print("创建Isaac Gym环境...")
        sim_params = get_sim_param()
        headless = not args.onscreen

        # 选择传给环境的 coacd 目录（如果存在）
        object_asset_dir = asset_dir_to_use if asset_dir_to_use else (mesh_dir if has_coacd else None)

        # 使用本文件中定义的 EvalIsaacEnv（基于原 _EvalIsaac 重构）
        env = EvalIsaacEnv(
            cfg,
            sim_params,
            gymapi.SIM_PHYSX,
            device,
            0,
            headless=headless,
            init_opt_q=opt_q,
            object_name=object_name,
            object_volume=object_volume,
            fix_object=False,
            enable_recording=enable_recording,
            record_dir=args.record_dir or args.output_dir,
            object_scale=scale,
            object_asset_dir=object_asset_dir,
            debug=args.debug,
        )
        log_print("环境创建成功")
        
        # 运行稳定性测试
        log_print("开始运行稳定性测试...")
        succ_all, succ_any = env.push_object()
        
        # 计算结果
        total = int(succ_all.shape[0])
        succ_cnt_all = int(succ_all.sum().item())
        succ_cnt_any = int(succ_any.sum().item())
        
        results = {
            'total': total,
            'succ_all': succ_cnt_all,
            'succ_any': succ_cnt_any,
            'rate_all': succ_cnt_all / total if total > 0 else 0.0,
            'rate_any': succ_cnt_any / total if total > 0 else 0.0,
        }
        render_result_summary(results)
        
        log_print(f"测试完成: total={total}, succ_all={succ_cnt_all}, succ_any={succ_cnt_any}")
        log_print(f"成功率(all): {results['rate_all']*100:.2f}%")
        log_print(f"成功率(any): {results['rate_any']*100:.2f}%")
        
        # 保存结果
        save_results(args.output_dir, task_spec, results, timestamp)
        
        # 保存可视化
        if task_spec.get('record_options', {}).get('save_visualization', True):
            save_visualization(args.output_dir, task_spec, qpos_batch, mesh_path, scale, timestamp, device)
        
        # 为纯手部静帧准备Pose（使用HTML同款抓取）
        try:
            env.static_hand_pose = env.q_transfer_o2s(opt_q[0:1]).clone().detach()
        except Exception:
            env.static_hand_pose = None

        # 静态预览模式：仅录制静态视频后退出
        if args.static_preview:
            if not enable_recording:
                env.enable_recording = True
                env._setup_cameras()
            record_dir = args.record_dir or args.output_dir
            vis_dir = os.path.join(record_dir, timestamp)
            os.makedirs(vis_dir, exist_ok=True)

            static_video = os.path.join(vis_dir, f"{timestamp}_{object_name}_static.mp4")
            env.record_static_scene(static_video, num_frames=150, fps=30, include_object=True)

            if task_spec.get('record_options', {}).get('save_hand_only', False):
                hand_only_filename = f"{timestamp}_{object_name}_hand_only.mp4"
                hand_only_path = os.path.join(vis_dir, hand_only_filename)
                env.record_hand_only(hand_only_path, num_frames=60, fps=30)

            log_print("静态预览完成，跳过稳定性仿真。", style="cyan")
            if hasattr(env, 'close'):
                env.close()
            env = None
            log_file.close()
            sys.exit(0)
        
        # 保存录制视频
        if enable_recording and hasattr(env, 'save_video'):
            record_dir = args.record_dir or args.output_dir
            vis_dir = os.path.join(record_dir, timestamp)
            os.makedirs(vis_dir, exist_ok=True)
            
            video_filename = f"{timestamp}_{object_name}.mp4"
            video_path = os.path.join(vis_dir, video_filename)
            env.save_video(video_path, fps=30)
            
            if task_spec.get('record_options', {}).get('save_hand_only', False):
                hand_only_filename = f"{timestamp}_{object_name}_hand_only.mp4"
                hand_only_path = os.path.join(vis_dir, hand_only_filename)
                env.record_hand_only(hand_only_path, num_frames=60, fps=30)
        
        # 清理环境（一次性最小清理，避免重复销毁仿真句柄）
        log_print("清理环境...", style="cyan")
        if hasattr(env, 'close'):
            env.close()
        env = None

        log_print("========== 任务成功完成 ==========", style="bold green", panel_title="任务状态")
        log_file.close()
        sys.exit(0)
        
    except Exception as e:
        log_print(f"========== 任务失败 ==========", style="bold red", panel_title="任务状态")
        log_print(f"错误: {str(e)}")
        log_print(traceback.format_exc())
        
        # 保存错误信息
        error_info = {
            'task_id': task_spec.get('task_id', 'unknown') if 'task_spec' in locals() else 'unknown',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': timestamp,
        }
        error_path = os.path.join(args.output_dir, 'error.json')
        with open(error_path, 'w') as f:
            json.dump(error_info, f, indent=2)
        
        log_file.close()
        sys.exit(1)
    finally:
        if temp_asset_dir is not None:
            try:
                shutil.rmtree(temp_asset_dir, ignore_errors=True)
            except Exception:
                pass


if __name__ == '__main__':
    main()
