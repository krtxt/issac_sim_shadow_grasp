"""
批量Isaac Gym稳定性测试调度脚本

职责：读取DexGraspNet数据、拆分任务、生成task_spec、GPU调度、进程监控、清理失败任务
"""
import os
import sys
import subprocess
import random
import time
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum
import tyro
import logging
import signal
import psutil
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rot6d import rot_to_orthod6d

# 默认元数据文件
DEFAULT_METADATA_FILES = [
    # 'debug_grasp_data_256.pt',
    # 'debug_grasp_data_128.pt',
    'dexgraspnet_shadowhand_downsample.pt',
    # 'dexgraspnet_shadowhand.pt',
]


console = Console()


@dataclass
class Config:
    """稳定性评估的配置参数"""
    # 数据路径
    dataset_path: str = '/home/xiantuo/source/grasp/GithubClone/SceneLeapUltra/data/DexGraspNet'
    object_root: Optional[str] = None  # 物体urdf/obj根目录，None则使用dataset_path/meshdata
    metadata_file: Optional[str] = None  # 可选，覆盖默认metadata文件
    
    # 输出路径
    output_root: str = 'outputs/stability_eval/test_record_all'
    # task_spec_root: str = 'outputs/stability_eval/task_specs'
    task_spec_root: str = os.path.join(output_root, 'task_specs')
    # log_dir: str = 'outputs/stability_eval/logs'
    log_dir: str = os.path.join(output_root, 'logs')
    
    # 数据/任务配置
    max_objects: int = -1 
    max_grasps_per_object: int = -1 
    split: str = 'all'  # 'train', 'test', or 'all'
    
    # 调度配置
    # gpu_ids: List[int] = field(default_factory=lambda: [0])
    gpu_ids: List[int] = field(default_factory=lambda: [0,1,2,4,5,6,7])
    jobs_per_gpu: int = 1
    timeout_duration: int = 600  # 每个任务超时时间(秒)
    
    # 录制选项
    enable_recording: bool = False
    save_hand_only_video: bool = False
    save_visualization: bool = False
    
    # 行为
    debug: bool = True
    onscreen: bool = False
    cpu: bool = False
    static_preview: bool = False
    
    def __post_init__(self):
        """初始化路径"""
        if self.object_root is None:
            self.object_root = os.path.join(self.dataset_path, 'meshdata')
        
        # 创建必要目录
        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(self.task_spec_root, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 验证参数
        if not self.gpu_ids:
            raise ValueError("gpu_ids不能为空")
        if self.jobs_per_gpu < 1:
            raise ValueError("jobs_per_gpu必须大于0")
        if self.timeout_duration < 1:
            raise ValueError("timeout_duration必须大于0")


class TaskStatus(Enum):
    """任务状态"""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"


def render_intro(config: "Config"):
    """使用 Rich 展示任务配置概览"""
    console.rule("[bold cyan]Isaac Gym 稳定性评估批量调度")
    info_table = Table(show_header=False, box=box.SIMPLE_HEAVY, expand=True)
    info_table.add_row("数据集路径", config.dataset_path)
    info_table.add_row("输出路径", config.output_root)
    info_table.add_row(
        "GPU 调度",
        f"IDs: {config.gpu_ids} / 每GPU任务数: {config.jobs_per_gpu}",
    )
    info_table.add_row(
        "物体/抓取上限",
        f"物体 {config.max_objects} · 每物体抓取 {config.max_grasps_per_object}",
    )
    info_table.add_row(
        "录制",
        f"视频:{'开' if config.enable_recording else '关'} · 手部:{'开' if config.save_hand_only_video else '关'} · 可视化:{'开' if config.save_visualization else '关'}",
    )
    console.print(info_table)
    console.rule()


def render_progress_panel(
    successful: int,
    total: int,
    attempt: int,
    timeout: Optional[int] = None,
    error: Optional[int] = None,
    title: str = "进度更新",
    style: str = "cyan",
):
    """渲染带统计信息的进度面板"""
    table = Table(show_header=False, box=box.SIMPLE, expand=False)
    table.add_row("成功任务", f"{successful}/{total}")
    table.add_row("总尝试", str(attempt))
    if timeout is not None:
        table.add_row("超时任务", str(timeout))
    if error is not None:
        table.add_row("错误任务", str(error))
    console.print(Panel.fit(table, title=title, border_style=style))


def render_final_summary(total: int, attempt: int, stats: dict):
    """渲染最终统计面板"""
    success = stats.get("successful_tasks", 0)
    timeout = stats.get("timeout_tasks", 0)
    error = stats.get("error_tasks", 0)
    if error > 0:
        border = "red"
    elif timeout > 0:
        border = "yellow"
    else:
        border = "green"

    summary_table = Table(show_header=False, box=box.SIMPLE_HEAVY, expand=False)
    summary_table.add_row("总任务", str(total))
    summary_table.add_row("成功", str(success))
    summary_table.add_row("超时", str(timeout))
    summary_table.add_row("错误", str(error))
    summary_table.add_row("总尝试", str(attempt))

    console.print(Panel(summary_table, title="评估完成统计", border_style=border))


class TaskLogger:
    """任务日志记录器"""
    def __init__(self, log_dir: str):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = log_dir

        self.logger = logging.getLogger("StabilityEval")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()

        # 详细日志文件
        log_file = os.path.join(log_dir, f"stability_eval_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(file_handler)

        # 统计信息
        self.stats_file = os.path.join(log_dir, f"task_stats_{timestamp}.json")
        self.stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "timeout_tasks": 0,
            "error_tasks": 0,
            "task_details": []
        }
        
        self.successful_task_ids = set()
        self.console = console

    def _print_task_start(self, task_id: str, gpu_id: int):
        message = f"分配 GPU {gpu_id}"
        panel_title = f"🚀 启动 {task_id}"
        self.console.print(Panel.fit(message, title=panel_title, border_style="cyan"))

    def _print_task_result(
        self,
        status: TaskStatus,
        task_id: str,
        duration: float,
        folder: str,
        gpu_id: int,
        error_msg: Optional[str] = None,
    ):
        style_map = {
            TaskStatus.SUCCESS: "green",
            TaskStatus.TIMEOUT: "yellow",
            TaskStatus.ERROR: "red",
        }
        icon_map = {
            TaskStatus.SUCCESS: "✅",
            TaskStatus.TIMEOUT: "⏰",
            TaskStatus.ERROR: "❌",
        }
        title_map = {
            TaskStatus.SUCCESS: "任务成功",
            TaskStatus.TIMEOUT: "任务超时",
            TaskStatus.ERROR: "任务失败",
        }

        details = [f"耗时 {duration:.2f}s", f"GPU {gpu_id}", f"目录 {folder}"]
        if error_msg:
            details.append(f"原因: {error_msg}")
        detail_text = "\n".join(details)
        panel_message = f"{icon_map[status]} [bold]{task_id}[/bold]\n{detail_text}"
        self.console.print(
            Panel.fit(panel_message, title=title_map[status], border_style=style_map[status])
        )

    def log_task_start(self, task_id: str, gpu_id: int):
        """记录任务开始"""
        self.logger.info(f"[{task_id}] 开始任务 on GPU {gpu_id}")
        self.stats["total_tasks"] += 1
        self._print_task_start(task_id, gpu_id)

    def log_task_end(self, task_id: str, status: TaskStatus, duration: float,
                     folder: str, gpu_id: int, error_msg: Optional[str] = None):
        """记录任务结束"""
        if status == TaskStatus.SUCCESS:
            self.logger.info(f"[{task_id}] 任务成功完成 in {duration:.2f}s")
            if task_id not in self.successful_task_ids:
                self.stats["successful_tasks"] += 1
                self.successful_task_ids.add(task_id)
        elif status == TaskStatus.TIMEOUT:
            self.logger.warning(f"[{task_id}] 任务超时 after {duration:.2f}s")
            self.stats["timeout_tasks"] += 1
        else:
            error_details = f": {error_msg}" if error_msg else ""
            self.logger.error(f"[{task_id}] 任务失败 after {duration:.2f}s{error_details}")
            self.stats["error_tasks"] += 1

        task_detail = {
            "task_id": task_id,
            "status": status.value,
            "duration": duration,
            "folder": folder,
            "gpu_id": gpu_id,
        }
        if error_msg:
            task_detail["error_message"] = error_msg

        self.stats["task_details"].append(task_detail)
        self._save_stats()
        self._print_task_result(status, task_id, duration, folder, gpu_id, error_msg)
    
    def _save_stats(self):
        """保存统计信息"""
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)


class GPUManager:
    """GPU资源管理器"""
    def __init__(self, gpu_ids: List[int], jobs_per_gpu: int):
        self.gpu_ids = gpu_ids
        self.jobs_per_gpu = jobs_per_gpu
        self.gpu_job_counts = {gpu_id: 0 for gpu_id in gpu_ids}
    
    def get_available_gpu(self) -> int:
        """返回当前负载最小的GPU ID"""
        return min(self.gpu_job_counts.items(), key=lambda x: x[1])[0]
    
    def add_job(self, gpu_id: int):
        """为指定GPU添加一个任务"""
        self.gpu_job_counts[gpu_id] += 1
    
    def remove_job(self, gpu_id: int):
        """为指定GPU移除一个任务"""
        self.gpu_job_counts[gpu_id] = max(0, self.gpu_job_counts[gpu_id] - 1)
    
    @property
    def total_jobs(self) -> int:
        """返回当前总任务数"""
        return sum(self.gpu_job_counts.values())
    
    @property
    def max_parallel_jobs(self) -> int:
        """返回最大并行任务数"""
        return len(self.gpu_ids) * self.jobs_per_gpu


class ProcessManager:
    """进程管理器"""
    def __init__(self):
        self.active_processes = []
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)
    
    def add_process(self, process, start_time, gpu_id, task_id, folder):
        """添加新进程"""
        self.active_processes.append((process, start_time, gpu_id, task_id, folder))
    
    def update_processes_status(
        self, 
        gpu_manager: GPUManager, 
        task_logger: TaskLogger, 
        config: Config, 
        processed_task_ids: set
    ) -> int:
        """检查所有活动进程的状态"""
        new_active_processes = []
        newly_successful_tasks = 0
        current_time = time.time()

        for proc, start_time, gpu_id, task_id, folder in self.active_processes:
            elapsed_time = current_time - start_time
            folder_path = os.path.join(config.output_root, folder)

            try:
                # 检查进程是否存在
                if not psutil.pid_exists(proc.pid):
                    gpu_manager.remove_job(gpu_id)
                    cleanup_task_folder(folder_path)
                    task_logger.log_task_end(task_id, TaskStatus.ERROR, elapsed_time, folder, gpu_id, 
                                           "进程意外终止")
                    continue

                # 检查超时
                if elapsed_time > config.timeout_duration:
                    parent = psutil.Process(proc.pid)
                    children = parent.children(recursive=True)
                    for child in children:
                        child.kill()
                    parent.kill()
                    
                    gpu_manager.remove_job(gpu_id)
                    cleanup_task_folder(folder_path)
                    task_logger.log_task_end(task_id, TaskStatus.TIMEOUT, elapsed_time, folder, gpu_id)
                    continue

                # 检查进程状态
                if proc.poll() is not None:
                    gpu_manager.remove_job(gpu_id)
                    
                    if task_id not in processed_task_ids:
                        processed_task_ids.add(task_id)
                        
                        if proc.returncode == 0:
                            # 正常退出，直接视为成功
                            task_logger.log_task_end(task_id, TaskStatus.SUCCESS, elapsed_time, folder, gpu_id)
                            newly_successful_tasks += 1
                        else:
                            # 非零退出码：先检查是否已成功产出 metrics.json
                            metrics_path = os.path.join(folder_path, "metrics.json")
                            has_valid_metrics = False
                            if os.path.exists(metrics_path):
                                try:
                                    with open(metrics_path, "r") as f:
                                        json.load(f)
                                    has_valid_metrics = True
                                except Exception:
                                    has_valid_metrics = False

                            if has_valid_metrics:
                                # 仿真流程已完成且结果文件存在，只是在退出阶段发生段错误等，视为成功
                                task_logger.log_task_end(task_id, TaskStatus.SUCCESS, elapsed_time, folder, gpu_id)
                                newly_successful_tasks += 1
                            else:
                                # 没有有效结果文件，才视为真正失败并清理目录
                                cleanup_task_folder(folder_path)
                                task_logger.log_task_end(
                                    task_id,
                                    TaskStatus.ERROR,
                                    elapsed_time,
                                    folder,
                                    gpu_id,
                                    f"退出码: {proc.returncode}",
                                )
                else:
                    new_active_processes.append((proc, start_time, gpu_id, task_id, folder))
                    
            except psutil.NoSuchProcess:
                gpu_manager.remove_job(gpu_id)
                cleanup_task_folder(folder_path)
                task_logger.log_task_end(task_id, TaskStatus.ERROR, elapsed_time, folder, gpu_id,
                                       "进程消失")

        self.active_processes = new_active_processes
        return newly_successful_tasks

    def handle_interrupt(self, signum, frame):
        """处理中断信号"""
        console.print("\n[bold yellow]正在清理进程...[/]")
        try:
            self.cleanup_all_processes()
        except Exception as e:
            console.print(f"[bold red]清理进程时发生错误:[/] {e}")
        finally:
            console.print("[bold green]清理完成[/]")
            exit(0)
    
    def cleanup_all_processes(self):
        """清理所有进程"""
        for proc, _, gpu_id, task_id, folder in self.active_processes:
            try:
                if psutil.pid_exists(proc.pid):
                    parent = psutil.Process(proc.pid)
                    timeout = 5
                    start_time = time.time()
                    
                    parent.terminate()
                    
                    while time.time() - start_time < timeout:
                        if not parent.is_running():
                            break
                        time.sleep(0.1)
                    
                    if parent.is_running():
                        children = parent.children(recursive=True)
                        for child in children:
                            child.kill()
                        parent.kill()
                    
                    console.print(f"[yellow]已终止任务 {task_id}[/]")
            except psutil.NoSuchProcess:
                pass
            except Exception as e:
                console.print(f"[bold red]终止任务 {task_id} 时出错:[/] {e}")


def generate_unique_id():
    """生成唯一标识符"""
    return random.randint(0, 2**32-1)


def cleanup_task_folder(folder_path: str):
    """清理失败的任务文件夹"""
    try:
        if os.path.exists(folder_path):
            for root, dirs, files in os.walk(folder_path, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except Exception as e:
                        logging.warning(f"无法删除文件 {name}: {str(e)}")
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except Exception as e:
                        logging.warning(f"无法删除目录 {name}: {str(e)}")
            shutil.rmtree(folder_path)
            logging.info(f"成功清理文件夹: {folder_path}")
    except Exception as e:
        logging.error(f"清理文件夹 {folder_path} 失败: {str(e)}")


def aggregate_successful_results(config: Config, task_logger: TaskLogger):
    """聚合所有成功任务的结果并清理单任务输出目录"""
    successful_ids = sorted(task_logger.successful_task_ids)
    if not successful_ids:
        print("没有成功完成的任务，无需汇总。")
        return

    html_dir = os.path.join(config.output_root, "html")
    video_dir = os.path.join(config.output_root, "videos")
    hand_video_dir = os.path.join(config.output_root, "hand_only_videos")
    metrics_summary_path = os.path.join(config.output_root, "metrics_summary.json")

    output_dirs = [html_dir, video_dir]
    if config.save_hand_only_video:
        output_dirs.append(hand_video_dir)

    for path in output_dirs:
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    aggregated_metrics: List[Dict[str, Any]] = []
    processed_task_dirs = []

    for task_id in successful_ids:
        folder_name = f"task_{task_id}"
        folder_path = os.path.join(config.output_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        metrics_path = os.path.join(folder_path, "metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                aggregated_metrics.append(metrics)
            except Exception as exc:
                logging.error(f"读取 {metrics_path} 失败: {exc}")

        for root, _, files in os.walk(folder_path):
            for filename in files:
                src = os.path.join(root, filename)
                lower = filename.lower()
                dest_name = f"{task_id}_{filename}"

                try:
                    if lower.endswith(".html"):
                        shutil.copy2(src, os.path.join(html_dir, dest_name))
                    elif lower.endswith(".mp4"):
                        if "hand_only" in lower:
                            if config.save_hand_only_video:
                                shutil.copy2(src, os.path.join(hand_video_dir, dest_name))
                        else:
                            shutil.copy2(src, os.path.join(video_dir, dest_name))
                except Exception as exc:
                    logging.error(f"复制文件 {src} 失败: {exc}")

        processed_task_dirs.append(folder_path)

    if aggregated_metrics:
        try:
            with open(metrics_summary_path, "w") as f:
                json.dump(aggregated_metrics, f, indent=2, ensure_ascii=False)
            print(f"已生成指标汇总文件: {metrics_summary_path}")
        except Exception as exc:
            logging.error(f"写入指标汇总失败: {exc}")

    for folder_path in processed_task_dirs:
        try:
            shutil.rmtree(folder_path)
            print(f"已清理任务目录: {folder_path}")
        except Exception as exc:
            logging.error(f"删除任务目录 {folder_path} 失败: {exc}")


def load_dexgraspnet_split(dataset_path, split='test'):
    """加载DexGraspNet数据集分割"""
    split_path = os.path.join(dataset_path, 'grasp.json')
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"缺少 grasp.json: {split_path}")
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    key_map = {
        'train': '_train_split',
        'test': '_test_split',
        'all': '_all_split'
    }
    split_key = key_map.get(split, split)
    if split_key not in split_data:
        raise KeyError(f"grasp.json 中缺少 {split_key}")
    return split_data[split_key]


def find_metadata_file(dataset_path, metadata_file=None):
    """查找metadata文件"""
    if metadata_file:
        candidate = Path(metadata_file)
        if not candidate.is_absolute():
            candidate = Path(dataset_path) / candidate
        if candidate.exists():
            return str(candidate.resolve())
        raise FileNotFoundError(f"指定的 metadata 文件不存在: {candidate}")

    dataset_dir = Path(dataset_path)
    for filename in DEFAULT_METADATA_FILES:
        candidate = dataset_dir / filename
        if candidate.exists():
            return str(candidate.resolve())
    raise FileNotFoundError(
        f"在 {dataset_dir} 中找不到有效的 metadata 文件，"
        f"需要其中之一: {DEFAULT_METADATA_FILES}"
    )


def load_dexgraspnet_gt(dataset_path, split='test', metadata_file=None):
    """加载DexGraspNet ground truth数据"""
    split_objects = load_dexgraspnet_split(dataset_path, split)
    ordered = OrderedDict((obj, []) for obj in split_objects)
    pt_path = find_metadata_file(dataset_path, metadata_file)
    console.log(f"使用 DexGraspNet metadata: {pt_path}")
    grasp_dataset = torch.load(pt_path, map_location='cpu')
    metadata = grasp_dataset.get('metadata', [])
    for mdata in metadata:
        obj_name = mdata['object_name']
        if obj_name not in ordered:
            continue
        hand_rot_mat = mdata['rotations'].clone().detach().cpu().float()
        joint = mdata['joint_positions'].clone().detach().cpu().float()
        trans = mdata['translations'].clone().detach().cpu().float()
        scale = float(mdata['scale'])
        rot6d = rot_to_orthod6d(hand_rot_mat.unsqueeze(0)).squeeze(0)
        trans_world = torch.matmul(hand_rot_mat, trans)
        qpos = torch.cat([trans_world, rot6d, joint], dim=0).numpy()
        ordered[obj_name].append({
            'qpos': qpos,
            'scale': scale,
        })
    return ordered


def create_task_spec(task_id, object_name, object_root, qpos_batch, scale_list, config):
    """创建任务配置文件"""
    task_spec = {
        'task_id': task_id,
        'object_name': object_name,
        'object_root': object_root,
        'qpos_batch': [qpos.tolist() if hasattr(qpos, 'tolist') else qpos for qpos in qpos_batch],
        'scale_list': scale_list,
        'sim_config': {
            'headless': not config.onscreen,
            'cpu': config.cpu,
        },
        'record_options': {
            'enable_recording': config.enable_recording,
            'save_hand_only': config.save_hand_only_video,
            'save_visualization': config.save_visualization,
        },
        'debug': config.debug,
    }
    
    task_spec_path = os.path.join(config.task_spec_root, f"{task_id}.json")
    with open(task_spec_path, 'w') as f:
        json.dump(task_spec, f, indent=2)
    
    return task_spec_path


def generate_task_command(config: Config, gpu_id: int, task_id: str, 
                          task_spec_path: str, task_output_dir: str) -> Tuple[List[str], dict]:
    """生成单个任务的命令"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    single_script = os.path.join(script_dir, "stability_eval_single.py")
    
    if not os.path.exists(single_script):
        raise FileNotFoundError(f"找不到单仿真脚本: {single_script}")
    
    cmd = [
        sys.executable,
        single_script,
        '--task_spec', task_spec_path,
        '--output_dir', task_output_dir,
    ]
    
    if config.onscreen:
        cmd.append('--onscreen')
    if config.cpu:
        cmd.append('--cpu')
    if config.debug:
        cmd.append('--debug')
    if config.static_preview:
        cmd.append('--static_preview')
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    return cmd, env


def main(config: Config):
    """主函数"""
    render_intro(config)
    
    # 创建日志和管理器
    task_logger = TaskLogger(config.log_dir)
    gpu_manager = GPUManager(config.gpu_ids, config.jobs_per_gpu)
    process_manager = ProcessManager()
    
    try:
        # 加载DexGraspNet数据
        console.log("加载DexGraspNet数据...")
        gt_data = load_dexgraspnet_gt(
            config.dataset_path,
            split=config.split,
            metadata_file=config.metadata_file,
        )
        num_objects_total = len(gt_data)
        num_objects_with_grasps = sum(1 for _name, grasps in gt_data.items() if len(grasps) > 0)
        console.print(
            Panel.fit(
                f"总物体 {num_objects_total}\n可用物体 {num_objects_with_grasps}",
                title="DexGraspNet 数据加载完成",
                border_style="green" if num_objects_with_grasps else "red",
            )
        )

        # 以 (object_name, scale) 作为最小单元进行分组
        # 每个分组对应一个物体在某个特定scale下的所有抓取
        grouped_by_obj_scale = {}
        for obj_name, grasps in gt_data.items():
            if len(grasps) == 0:
                continue
            for g in grasps:
                s = float(g["scale"])
                key = (obj_name, s)
                if key not in grouped_by_obj_scale:
                    grouped_by_obj_scale[key] = []
                grouped_by_obj_scale[key].append(g)

        num_groups_total = len(grouped_by_obj_scale)
        console.log(f"按 (object_name, scale) 分组得到 {num_groups_total} 个候选任务单元")

        # 准备任务列表：遍历所有 (object_name, scale) 分组，直到达到 max_objects
        max_objects_limit = config.max_objects if config.max_objects > 0 else None
        tasks = []
        task_id_counter = 0
        for (obj_name, scale), grasps in grouped_by_obj_scale.items():
            if len(grasps) == 0:
                continue

            # 截断到每个 (object, scale) 组内最多抓取数
            if config.max_grasps_per_object and config.max_grasps_per_object > 0:
                selected_grasps = grasps[: config.max_grasps_per_object]
            else:
                selected_grasps = grasps
            if len(selected_grasps) == 0:
                continue

            qpos_batch = [g["qpos"] for g in selected_grasps]
            # 该任务内所有抓取共享同一个scale
            scale_list = [float(scale)] * len(selected_grasps)

            task_id = f"{obj_name}_s{scale:.6f}_{task_id_counter:04d}"
            task_id_counter += 1

            tasks.append(
                {
                    "task_id": task_id,
                    "object_name": obj_name,
                    "qpos_batch": qpos_batch,
                    "scale_list": scale_list,
                }
            )

            if max_objects_limit is not None and len(tasks) >= max_objects_limit:
                break

        if len(tasks) == 0:
            console.print("[bold yellow]警告: 没有任何 (object, scale) 组合可用于测试[/]")
        else:
            console.print(
                Panel.fit(
                    f"共 {len(tasks)} 个任务 (限制 {config.max_objects})",
                    title="任务准备完成",
                    border_style="cyan",
                )
            )
        
        # 执行任务调度
        successful_tasks = 0
        attempt_num = 0
        processed_task_ids = set()
        task_queue = tasks.copy()
        
        while successful_tasks < len(tasks):
            # 更新进程状态
            newly_finished_count = process_manager.update_processes_status(
                gpu_manager, task_logger, config, processed_task_ids
            )
            if newly_finished_count > 0:
                successful_tasks += newly_finished_count
                render_progress_panel(
                    successful_tasks,
                    len(tasks),
                    attempt_num,
                    timeout=task_logger.stats["timeout_tasks"],
                    error=task_logger.stats["error_tasks"],
                    title="任务进度",
                    style="green",
                )

            # 启动新任务
            while gpu_manager.total_jobs < gpu_manager.max_parallel_jobs and task_queue:
                task_info = task_queue.pop(0)
                attempt_num += 1
                
                task_id = task_info['task_id']
                task_folder_name = f"task_{task_id}"
                task_output_dir = os.path.join(config.output_root, task_folder_name)
                
                # 创建task_spec文件
                task_spec_path = create_task_spec(
                    task_id,
                    task_info['object_name'],
                    config.object_root,
                    task_info['qpos_batch'],
                    task_info['scale_list'],
                    config
                )
                
                # 生成命令
                gpu_id = gpu_manager.get_available_gpu()
                gpu_manager.add_job(gpu_id)
                
                cmd, env = generate_task_command(config, gpu_id, task_id, 
                                                task_spec_path, task_output_dir)
                task_logger.log_task_start(task_id, gpu_id)
                
                # 启动进程
                process = subprocess.Popen(cmd, env=env)
                time.sleep(0.5)
                
                if process.poll() is None:
                    os.makedirs(task_output_dir, exist_ok=True)
                    process_manager.add_process(process, time.time(), gpu_id, 
                                              task_id, task_folder_name)
                else:
                    gpu_manager.remove_job(gpu_id)
                    task_logger.log_task_end(task_id, TaskStatus.ERROR, 0.5, 
                                           task_folder_name, gpu_id,
                                           f"进程启动失败: {process.returncode}")
            
            # 定期保存统计
            if attempt_num > 0 and attempt_num % 10 == 0:
                task_logger._save_stats()
                render_progress_panel(
                    successful_tasks,
                    len(tasks),
                    attempt_num,
                    timeout=task_logger.stats["timeout_tasks"],
                    error=task_logger.stats["error_tasks"],
                    title="阶段汇报",
                    style="magenta",
                )
            
            # 等待
            if (gpu_manager.total_jobs >= gpu_manager.max_parallel_jobs or
               (successful_tasks + gpu_manager.total_jobs >= len(tasks) and 
                process_manager.active_processes)):
                time.sleep(1)
            elif not process_manager.active_processes and successful_tasks < len(tasks):
                time.sleep(0.1)

        # 等待所有剩余进程完成
        while process_manager.active_processes:
            newly_finished_count = process_manager.update_processes_status(
                gpu_manager, task_logger, config, processed_task_ids
            )
            if newly_finished_count > 0:
                successful_tasks += newly_finished_count
                render_progress_panel(
                    successful_tasks,
                    len(tasks),
                    attempt_num,
                    timeout=task_logger.stats["timeout_tasks"],
                    error=task_logger.stats["error_tasks"],
                    title="收尾任务进度",
                    style="blue",
                )
            time.sleep(1)
        
        aggregate_successful_results(config, task_logger)
            
    except KeyboardInterrupt:
        console.print("\n[bold yellow]接收到中断信号，正在清理...[/]")
        process_manager.cleanup_all_processes()
    finally:
        task_logger._save_stats()
        render_final_summary(len(tasks), attempt_num, task_logger.stats)


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
