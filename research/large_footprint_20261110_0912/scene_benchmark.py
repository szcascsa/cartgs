"""原冻结测试的单场景包装；不改变算法/条件/样本数，仅隔离外部干扰。"""
import sys
import paired_benchmark as benchmark

index=sys.argv.index('--scene');slug=sys.argv[index+1];del sys.argv[index:index+2]
benchmark.SCENES=[s for s in benchmark.SCENES if s['slug']==slug]
assert len(benchmark.SCENES)==1
benchmark.main()
