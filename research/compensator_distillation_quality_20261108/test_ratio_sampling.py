"""Ratio 连续采样保持配对、可恢复并覆盖全部区间。"""
from common import *
from distill_compensation import training_ratio


def main():
    ratios = [.01, .05, .1, .3, .5, .7]
    jittered = 0
    intervals = set()
    for pair in range(1800):
        i = pair*2
        assert training_ratio(ratios, i, 2, 0) == ratios[pair % 6]
        value = training_ratio(ratios, i, 2, 0, .5)
        assert value == training_ratio(ratios, i+1, 2, 0, .5)
        assert ratios[0] <= value <= ratios[-1]
        if value not in ratios:
            jittered += 1
            intervals.add(next(j for j in range(5) if ratios[j] <= value <= ratios[j+1]))
        torch.rand(17)
        assert value == training_ratio(ratios, i, 2, 0, .5)
    assert 750 < jittered < 1050 and len(intervals) == 5
    save(ROOT/'reports/RATIO_SAMPLING_TEST.json', dict(passed=True, pairs=1800,
        jittered_pairs=jittered, intervals=sorted(intervals), global_local_ratio_paired=True,
        independent_of_global_rng=True, checkpoint_replay_by_seed_and_step=True))
    print('RATIO_SAMPLING_TEST_PASS', jittered, flush=True)


if __name__ == '__main__':
    main()
