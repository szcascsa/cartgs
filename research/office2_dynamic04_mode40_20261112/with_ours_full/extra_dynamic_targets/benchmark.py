"""Run the existing paired offline-search protocol with another latency target."""
from pathlib import Path
import os

root = Path(__file__).resolve().parent
target = float(os.environ['DYNAMIC_TARGET_MS'])
assert target in (0.5, 0.7)
prior = root.parent.parent/'benchmark.py'
source = prior.read_text(encoding='utf-8')
source = source.replace("ROOT = Path(__file__).resolve().parent", "ROOT = Path(__file__).resolve().parent.parent.parent")
source = source.replace("run = 'smoke' if args.smoke else 'formal'", "run = 'with_ours_full/extra_dynamic_targets/' + ('target05' if TARGET == .5 else 'target07') + ('/smoke' if args.smoke else '/formal')")
source = source.replace("TARGET = .4", f"TARGET = {target}")
source = source.replace("target_band_ms=[.38, .42]", "target_band_ms=[TARGET-TOLERANCE, TARGET+TOLERANCE]")
source = source.replace("to 0.4ms;", f"to {target}ms;")
source = source.replace("monitor = ROOT/'logs'/(run+'_MONITOR.jsonl')", "monitor = output.parent/'logs'/((('smoke' if args.smoke else 'formal'))+'_MONITOR.jsonl')")
exec(compile(source, str(prior), 'exec'), {'__name__': '__main__', '__file__': str(root/'runner_entry.py')})
