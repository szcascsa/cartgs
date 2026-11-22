"""复用已验证的 NVTX/correlation 归属逻辑，读取本轮独立 capture。"""
from pathlib import Path

dest = Path(__file__).resolve().parent
source = (dest.parent/'cuda_selector_implementation_20261015/parse_nsight.py').read_text()
source = source.replace('root = Path(__file__).resolve().parent', f'root = Path({str(dest / "final_measurement")!r})')
source = source.replace("prefix = sys.argv[1] if len(sys.argv)>1 else ''", "prefix = ''")
exec(compile(source, __file__, 'exec'))
