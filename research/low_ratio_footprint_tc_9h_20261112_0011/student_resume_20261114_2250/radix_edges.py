"""严格K和稳定顺序：小候选集排序收尾与冻结Radix逐位配对。"""
from student_study import *
from radix_runtime import extension


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--sanitizer', action='store_true')
    args = parser.parse_args()
    out = HERE / 'reports' / args.output
    out.mkdir(exist_ok=False)
    module = extension()
    torch.manual_seed(1789)
    checks = []
    sizes = [1, 31, 32, 33, 127, 255, 256, 257, 511, 1024, 4095, 4096, 16383, 16384, 20993, 65537]
    if args.sanitizer:
        sizes = [31, 33, 255, 257, 4096, 20993]
    retained = []
    for n in sizes:
        rows = torch.randperm(n, device='cuda')
        eligible = torch.rand(n, device='cuda') > .13
        counts = int(eligible.sum())
        scratch = [torch.empty(n, dtype=torch.int32, device='cuda'), torch.empty(n, dtype=torch.int32, device='cuda'),
                   torch.empty(264, dtype=torch.int32, device='cuda'),
                   torch.empty(2 * ((n+255)//256), dtype=torch.int32, device='cuda')]
        status = torch.zeros(1, dtype=torch.int32, device='cuda')
        for kind in ['random', 'ties', 'equal', 'saturated', 'signed_zero', 'negative']:
            scores = torch.rand(n, device='cuda')
            if kind == 'ties':
                scores = (scores * 16).floor() / 16
            elif kind == 'equal':
                scores.fill_(.5)
            elif kind == 'saturated':
                scores = 1. - scores * 1e-6
            elif kind == 'signed_zero':
                scores = torch.zeros(n, device='cuda')
                scores[::2] = -0.
            elif kind == 'negative':
                scores = scores * 4 - 2
            valid = eligible[rows]
            ranking = torch.argsort(scores.masked_fill(~valid, -float('inf')), descending=True, stable=True)
            for k in sorted({0, min(1, counts), min(31, counts), min(256, counts), counts//10, counts//2, counts}):
                selected = torch.zeros(n, dtype=torch.bool, device='cuda')
                selected[ranking[:k]] = True
                expected = rows[selected]
                for name in ['_radix_fused', '_radix_small']:
                    actual = getattr(module, name)(scores, rows, eligible, k, scratch, status)
                    row = dict(n=n, K=k, kind=kind, variant=name, passed=torch.equal(actual, expected))
                    checks.append(row)
                    assert row['passed'], (row, actual[:20], expected[:20])
                    retained.append((actual, actual.clone()))
                if len(retained)>32:
                    assert all(torch.equal(a,b) for a,b in retained)
                    retained.clear()
    assert all(torch.equal(a,b) for a,b in retained)
    append(out/'checks.csv', checks)
    save(out/'DONE.json', dict(passed=True, checks=len(checks), sanitizer=args.sanitizer))
    print('RADIX_FINISH_EDGES_COMPLETE', len(checks), flush=True)


if __name__ == '__main__':
    main()
