"""与 include/mixed_ratio.h 对应：独立随机流、截断高斯分位及等效ratio校准。"""
import math


class Random:
    def __init__(self, seed, sample):
        self.state = (seed ^ ((sample+1)*0xd1b54a32d192ed03)) & ((1 << 64)-1)

    def uniform(self):
        mask = (1 << 64)-1
        self.state = (self.state+0x9e3779b97f4a7c15) & mask
        z = self.state
        z = ((z ^ (z >> 30))*0xbf58476d1ce4e5b9) & mask
        z = ((z ^ (z >> 27))*0x94d049bb133111eb) & mask
        z ^= z >> 31
        return ((z >> 12)+.5)/4503599627370496.


def draw(target, low, high, seed=0, sample=0, concentration=(8, 12, 20), eta_min=.3, eta_max=.8):
    if (not math.isfinite(target) or not 0 < low < high <= 1 or not low <= target <= high or
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or
            isinstance(sample, bool) or not isinstance(sample, int) or not 0 <= sample < 2**53 or
            len(concentration) != 3 or any(isinstance(n, bool) or not isinstance(n, int) or not 0 < n <= 1000 for n in concentration) or
            not 0 < eta_min <= eta_max <= 1):
        raise ValueError('Invalid mixed ratio target, seed, sample or distribution')
    rng = Random(seed, sample)
    gamma = [-sum(math.log(rng.uniform()) for _ in range(n)) for n in concentration]
    shares = tuple(x/sum(gamma) for x in gamma)
    sigma = (eta_min+(eta_max-eta_min)*rng.uniform())*min(target-low, high-target)
    quantiles = tuple(sorted((rng.uniform() for _ in range(3)), reverse=True))
    return dict(shares=shares, sigma=sigma, quantiles=quantiles)


def normal_mass(a, b):
    root = math.sqrt(2.)
    if a >= 0:return .5*(math.erfc(a/root)-math.erfc(b/root))
    if b <= 0:return .5*(math.erfc(-b/root)-math.erfc(-a/root))
    return 1.-.5*math.erfc(b/root)-.5*math.erfc(-a/root)


def quantile(mu, sigma, q, low, high):
    left, right = low, high
    a = (low-mu)/sigma
    mass = normal_mass(a, (high-mu)/sigma)
    for _ in range(56):
        mid = (left+right)*.5
        if normal_mass(a, (mid-mu)/sigma) < q*mass:left = mid
        else:right = mid
    return (left+right)*.5


def calibrate(sample, counts, target, low, high):
    sigma, qs, n = sample['sigma'], sample['quantiles'], sum(counts)
    if sigma <= 0 or not n:return (target,)*3
    left, right = low-32*sigma, high+32*sigma
    for _ in range(64):
        mu = (left+right)*.5
        ratios = tuple(quantile(mu, sigma, q, low, high) for q in qs)
        if sum(c/n*r for c, r in zip(counts, ratios)) < target:left = mu
        else:right = mu
    return ratios


def integer_budgets(counts, ratios, target):
    """最大余数只分配取整误差；slot不足仍报告缺口，不跨档补点。"""
    real = [n*r for n, r in zip(counts, ratios)]
    budgets = [math.floor(k) for k in real]
    missing = math.floor(sum(counts)*target)-sum(budgets)
    for i in sorted(range(3), key=lambda i: (-(real[i]-budgets[i]), i))[:max(0, missing)]:
        budgets[i] += 1
    if missing < 0:
        for i in sorted((i for i in range(3) if budgets[i]), key=lambda i: (real[i]-budgets[i], i))[:-missing]:
            budgets[i] -= 1
    return budgets
