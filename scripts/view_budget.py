"""启动固定预算网页Viewer，默认融合CUDA推理；--viewer native可选原生界面。"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--budget', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--inference-backend', choices=('pytorch', 'cuda', 'cuda_graph'), default='cuda')
    parser.add_argument('--viewer', choices=('native', 'browser'), default='browser')
    parser.add_argument('--viewer-binary', type=Path, default=root/'bin/view_budget')
    parser.add_argument('--display', help='原Viewer已有X11会话，例如localhost:12.0')
    parser.add_argument('--port', type=int, default=0, help='0为自动选择空闲本机端口')
    args = parser.parse_args()
    if args.budget < 0 or not 0 <= args.port <= 65535:parser.error('预算或端口不合法')
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '2')
    if args.display:env['DISPLAY'] = args.display
    if args.viewer == 'native':
        if not args.viewer_binary.is_file():parser.error('原Viewer可执行文件不存在: ' + str(args.viewer_binary))
        if not env.get('DISPLAY'):parser.error('需要现有X11 DISPLAY；可用--display指定，或显式选择--viewer browser')
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', args.port))
        port = reservation.getsockname()[1]
    command = [sys.executable, str(root/'python/budget_viewer.py'), str(args.checkpoint),
               '--budget', str(args.budget), '--seed', str(args.seed), '--port', str(port),
               '--inference-backend', args.inference_backend]
    for name in ('dataset', 'config'):
        if getattr(args, name):command += ['--'+name, str(getattr(args, name))]
    with tempfile.TemporaryDirectory(prefix='cartgs_budget_') as temporary:
        state = Path(temporary)/'backend.json'
        backend = subprocess.Popen(command+['--state-file', str(state)], env=env)
        viewer = None
        try:
            started = time.monotonic()
            print('加载固定预算模型；网页为首选入口，推理后端请求='+args.inference_backend+'。', flush=True)
            while not state.exists():
                if backend.poll() is not None:raise RuntimeError('推理服务加载失败，exit='+str(backend.returncode))
                if time.monotonic()-started > 180:raise RuntimeError('推理服务启动超过180秒')
                time.sleep(.1)
            if json.loads(state.read_text())['pid'] != backend.pid:raise RuntimeError('推理服务PID不匹配')
            with urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as response:
                if not json.load(response)['ready']:raise RuntimeError('推理服务未就绪')
            if args.viewer == 'native':
                viewer = subprocess.Popen([str(args.viewer_binary), str(port)], env=env)
                return viewer.wait()
            print(f'网页 Viewer: http://127.0.0.1:{port}；远程使用 SSH 端口转发。Ctrl+C退出。', flush=True)
            print(f'端口转发示例: ssh -N -L {port}:127.0.0.1:{port} prmi@172.19.163.58', flush=True)
            return backend.wait()
        except KeyboardInterrupt:
            return 0
        finally:
            for process in (viewer, backend):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__ == '__main__':sys.exit(main())
