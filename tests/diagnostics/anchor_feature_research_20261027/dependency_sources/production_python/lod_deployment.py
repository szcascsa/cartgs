"""Post-mapping LoD：冻结 ranking、独立 deployment U、任意预算查询。"""
import hashlib
import json

import numpy as np
from scipy.special import expit


def array_sha(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_version(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def map_identity(row_ids, master_version, selector_version, topology_version, geometry_version):
    """调用者必须提交当前 map identity；不以点数相等代替 row/version 验证。"""
    rows = np.asarray(row_ids, dtype=np.int64)
    if rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows):
        raise ValueError("Gaussian row IDs must be a nonempty unique vector")
    if not master_version or not selector_version or not str(topology_version):
        raise ValueError("Master, Selector and topology versions are required")
    return dict(count=len(rows), row_mapping_sha=array_sha(rows), master_version=str(master_version),
                selector_version=str(selector_version), topology_version=str(topology_version), geometry_version=geometry_version)


def deployment_identity(model, position, rotation, scale, row_ids, master_version, topology_version):
    geometry = hashlib.sha256()
    for tensor in (position, rotation, scale):
        geometry.update(array_sha(tensor.detach().cpu().numpy()).encode())
    return map_identity(row_ids, master_version, model_version(model), topology_version, geometry.hexdigest())


def validate_ratio(ratio):
    ratio = float(ratio)
    if not np.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError("LoD ratio must be finite and strictly between 0 and 1")
    return ratio


def solve_bias_reference(score, ratio, tau=1.0):
    ratio = validate_ratio(ratio)
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 1 or not len(score) or not np.isfinite(score).all():
        raise ValueError("Ranking must be a nonempty finite vector")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("Temperature must be finite and positive")
    target = ratio * len(score)
    lo, hi = float(-score.max()-80*tau), float(-score.min()+80*tau)
    # 普通预算严格沿用 Strong-GO bracket；极端合法 r 扩展 bracket。
    while float(expit((score+lo)/tau).sum()) > target:
        lo -= max(80*tau, abs(lo))
    while float(expit((score+hi)/tau).sum()) < target:
        hi += max(80*tau, abs(hi))
    for iteration in range(1, 129):
        bias = (lo+hi)*0.5
        probability = expit((score+bias)/tau)
        expected = float(probability.sum(dtype=np.float64))
        residual = expected-target
        if abs(residual) <= 1e-7:
            return probability, dict(bias=bias, iterations=iteration, expected_k=expected,
                                     target_k=target, residual=residual)
        if residual > 0:
            hi = bias
        else:
            lo = bias
    raise RuntimeError("Analytic budget solver did not reach 1e-7 absolute tolerance")


# Phase 1 public helper remains the exact bisection reference.
solve_bias = solve_bias_reference


def solve_bias_fast(score, ratio, tau=1.0, *, initial_bias=None, bracket=None):
    """FP64 safeguarded Newton；初值只影响迭代数，不替代真实 budget equation。"""
    ratio = validate_ratio(ratio)
    a = np.asarray(score, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Ranking must be a nonempty finite vector")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("Temperature must be finite and positive")
    target = ratio*len(a)
    if min(target,len(a)-target) <= 1e-6 or ratio < expit(-80):
        # absolute tolerance尾端的root区间可很宽，局部F'不能界定reference bias。
        probability, root = solve_bias_reference(a,ratio,tau)
        root['solver'] = 'reference_tolerance_tail'
        return probability, root
    lo, hi = bracket if bracket is not None else (float(-a.max()-80*tau), float(-a.min()+80*tau))
    bias = float(initial_bias) if initial_bias is not None else (lo+hi)*.5
    if not np.isfinite(bias) or not lo < bias < hi:
        bias = (lo+hi)*.5
    previous_error = np.inf
    newton_steps = 0
    bisection_steps = 0
    for iteration in range(1,129):
        probability = expit((a+bias)/tau)
        expected = float(probability.sum(dtype=np.float64))
        residual = expected-target
        derivative = float((probability*(1-probability)).sum(dtype=np.float64)/tau)
        if abs(residual) <= 1e-7:
            if derivative <= 1e-12:
                probability, root = solve_bias_reference(a,ratio,tau)
                root['solver'] = 'reference_flat_derivative'
                return probability, root
            return probability, dict(bias=bias,iterations=iteration,expected_k=expected,target_k=target,
                residual=residual,derivative=derivative,solver='safeguarded_newton',
                newton_steps=newton_steps,bisection_steps=bisection_steps)
        if residual > 0:
            hi = bias
        else:
            lo = bias
        candidate = bias-residual/derivative if derivative > np.finfo(np.float64).tiny else np.nan
        if not np.isfinite(candidate) or not lo < candidate < hi or abs(residual) >= previous_error or candidate == bias:
            candidate = (lo+hi)*.5
            bisection_steps += 1
        else:
            newton_steps += 1
        previous_error = abs(residual)
        bias = candidate
    probability, root = solve_bias_reference(a, ratio, tau)
    root.update(solver='reference_convergence_fallback',newton_steps=newton_steps,bisection_steps=bisection_steps)
    return probability, root


class DeploymentState:
    """固定 final map 的不可变快照；训练 U 不属于本类。"""
    def __init__(self, score, deployment_uniform, row_ids, metadata, *, rank_order=None, activation_bias=None):
        self.score = np.array(score, dtype=np.float64, copy=True)
        self.deployment_uniform = np.array(deployment_uniform, dtype=np.float64, copy=True)
        self.row_ids = np.array(row_ids, dtype=np.int64, copy=True)
        if self.score.ndim != 1 or self.score.shape != self.deployment_uniform.shape or self.score.shape != self.row_ids.shape:
            raise ValueError("Deployment arrays are not row-aligned")
        if not np.isfinite(self.score).all() or not np.all((self.deployment_uniform >= 0) & (self.deployment_uniform < 1)):
            raise ValueError("Invalid deployment score or uniform")
        identity = map_identity(self.row_ids, metadata['identity']['master_version'],
                                metadata['identity']['selector_version'], metadata['identity']['topology_version'],
                                metadata['identity']['geometry_version'])
        if identity != metadata['identity'] or array_sha(self.score) != metadata['ranking_version']:
            raise ValueError("Deployment identity/ranking checksum mismatch")
        if array_sha(self.deployment_uniform) != metadata['uniform_sha']:
            raise ValueError("Deployment uniform checksum mismatch")
        version = {key:value for key,value in metadata.items() if key != 'deploy_version'}
        if hashlib.sha256(json.dumps(version, sort_keys=True).encode()).hexdigest() != metadata['deploy_version']:
            raise ValueError("Deployment metadata checksum mismatch")
        if metadata['format_version'] != 1 or metadata['architecture'] != 'wide_additive' or not np.isfinite(metadata['tau']) or metadata['tau'] <= 0:
            raise ValueError("Invalid deployment schema/temperature")
        self._metadata = json.loads(json.dumps(metadata))
        for value in (self.score, self.deployment_uniform, self.row_ids):
            value.setflags(write=False)
        self._initialize_runtime_cache(rank_order, activation_bias)

    def _initialize_runtime_cache(self, rank_order=None, activation_bias=None):
        n = len(self.score)
        if rank_order is None:
            rank_order = np.lexsort((self.row_ids, -self.score))
        self.rank_order = np.array(rank_order,dtype=np.int64,copy=True)
        if self.rank_order.shape != (n,) or np.any(self.rank_order < 0) or np.any(self.rank_order >= n) or not np.all(np.bincount(self.rank_order,minlength=n)==1):
            raise ValueError("Invalid cached rank permutation")
        ranked = self.score[self.rank_order]
        ids = self.row_ids[self.rank_order]
        if np.any(ranked[1:] > ranked[:-1]) or np.any((ranked[1:]==ranked[:-1]) & (ids[1:]<ids[:-1])):
            raise ValueError("Cached ranking does not match stable score/row-ID order")
        tau = self._metadata['tau']
        u = self.deployment_uniform
        with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
            derived = tau*(np.log(u)-np.log1p(-u))-self.score
            guard = 16*np.finfo(np.float64).eps*(np.abs(derived)+np.abs(self.score)+tau)
            guard += 4*tau*np.finfo(np.float64).eps/np.maximum(u*(1-u),np.finfo(np.float64).tiny)
        if activation_bias is not None:
            cached = np.asarray(activation_bias,dtype=np.float64)
            # log/log1p在不同libm平台可差几个ULP；缓存只作校验hint，始终本机重建。
            tolerance = 8*np.finfo(np.float64).eps*(np.abs(derived)+np.abs(self.score)+tau)
            if cached.shape!=derived.shape:
                raise ValueError("Cached activation bias does not match score/deployment U")
            with np.errstate(invalid='ignore'):
                matches = (cached==derived)|(np.isfinite(cached)&np.isfinite(derived)&
                    (np.abs(cached-derived)<=tolerance))
            if not np.all(matches):
                raise ValueError("Cached activation bias does not match score/deployment U")
        self.activation_bias = derived
        self._extreme_uniform_rows = np.flatnonzero((u==0)|(u<np.finfo(np.float64).tiny)|(u>1-1e-12))
        guard[self._extreme_uniform_rows] = 0
        self._activation_guard = guard
        self._bracket = (float(-self.score.max()-80*tau),float(-self.score.min()+80*tau))
        for value in (self.rank_order,self.activation_bias,self._activation_guard,self._extreme_uniform_rows):
            value.setflags(write=False)

    @property
    def cache_bytes(self):
        return dict(rank_order=self.rank_order.nbytes,activation_bias=self.activation_bias.nbytes,
                    numerical_guard=self._activation_guard.nbytes,extreme_rows=self._extreme_uniform_rows.nbytes)

    @classmethod
    def finalize(cls, model, position, rotation, scale, row_ids, *, master_version,
                 topology_version, seed=20261019, tau=1.0):
        if model.architecture != 'wide_additive':
            raise ValueError("arbitrary_lod requires WideAdditive checkpoint")
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Freeze/eval Selector before finalizing deployment state")
        if any(value.requires_grad for value in (position, rotation, scale)):
            raise ValueError("Freeze Master before finalizing deployment state")
        score = model.compute_ranking_score(position, rotation, scale).detach().cpu().numpy().astype(np.float64)
        identity = deployment_identity(model, position, rotation, scale, row_ids, master_version, topology_version)
        uniform = np.random.Generator(np.random.PCG64(seed)).random(len(score))
        metadata = dict(format_version=1, architecture='wide_additive', tau=float(tau), seed=int(seed),
                        rng='numpy.PCG64', identity=identity, ranking_version=array_sha(score),
                        uniform_sha=array_sha(uniform), lifecycle='post_mapping_frozen')
        metadata['deploy_version'] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        return cls(score, uniform, row_ids, metadata)

    @property
    def metadata(self):
        return json.loads(json.dumps(self._metadata))

    def validate(self, identity):
        if identity != self._metadata['identity']:
            raise ValueError("Stale deployment state: Master/Selector/topology/row mapping changed; finalize again")

    def query_lod(self, ratio, backend, *, identity):
        self.validate(identity)
        ratio = validate_ratio(ratio)
        if backend not in ('fresh_u', 'topk'):
            raise ValueError("LoD backend must be explicitly fresh_u or topk")
        mask = np.zeros(len(self.score), dtype=bool)
        result = dict(ratio=ratio, tf_ratio=ratio, backend=backend,
                      ranking_version=self._metadata['ranking_version'], deploy_version=self._metadata['deploy_version'])
        if backend == 'fresh_u':
            # 已缓存的 score quantile 只是 Newton 初值；不缓存 ratio 结果。
            rank_index = min(len(self.score)-1,int(ratio*len(self.score)))
            probability, root = solve_bias_fast(self.score, ratio, self._metadata['tau'],
                initial_bias=-self.score[self.rank_order[rank_index]],bracket=self._bracket)
            if root.get('solver','').startswith('reference'):
                mask[:] = self.deployment_uniform < probability
            else:
                bias = root['bias']
                # 接近activation边界时保留Phase 1的bisection/expit浮点约定。
                root_guard = 16e-7/root['derivative']
                near = np.abs(self.activation_bias-bias) <= self._activation_guard+root_guard
                if np.any(near):
                    probability, root = solve_bias_reference(self.score,ratio,self._metadata['tau'])
                    root['solver'] = 'reference_boundary_fallback'
                    mask[:] = self.deployment_uniform < probability
                else:
                    mask[:] = self.activation_bias < bias
                    rows = self._extreme_uniform_rows
                    mask[rows] = self.deployment_uniform[rows] < probability[rows]
            result.update(root)
            result['probability'] = probability
        else:
            count = int(np.floor(ratio*len(mask)+0.5))
            mask[self.rank_order[:count]] = True
            result['target_k'] = count
        indices = np.flatnonzero(mask)
        result.update(mask=mask, selected_indices=indices, selected_rows=self.row_ids[indices], actual_k=len(indices))
        return result

    def save(self, path):
        # 重复 export 应 load，禁止静默重建/覆盖 U。
        with open(path, 'xb') as stream:
            np.savez_compressed(stream, score=self.score, deployment_uniform=self.deployment_uniform,
                                row_ids=self.row_ids, metadata=json.dumps(self._metadata, sort_keys=True),
                                rank_order=self.rank_order,activation_bias=self.activation_bias,runtime_cache_version=1)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if 'runtime_cache_version' in data and int(data['runtime_cache_version']) != 1:
                raise ValueError("Unsupported deployment runtime cache version")
            return cls(data['score'], data['deployment_uniform'], data['row_ids'], json.loads(str(data['metadata'])),
                       rank_order=data['rank_order'] if 'rank_order' in data else None,
                       activation_bias=data['activation_bias'] if 'activation_bias' in data else None)


def save_selection_bundle(path, result):
    """仅 membership / TF ratio；不把 ratio-conditioned TF 属性声称为固定层。"""
    arrays = {key: result[key] for key in ('mask', 'selected_indices', 'selected_rows')}
    metadata = {key: value for key, value in result.items() if key not in arrays and key != 'probability'}
    np.savez_compressed(path, **arrays, metadata=json.dumps(metadata, sort_keys=True))


def incremental_layers(results):
    previous = np.zeros_like(results[0]['mask'])
    rows, offsets = [], [0]
    for result in results:
        mask = result['mask']
        if np.any(previous & ~mask):
            raise ValueError("LoD masks are not nested")
        added = np.flatnonzero(mask & ~previous)
        rows.extend(added.tolist()); offsets.append(len(rows))
        previous = mask
    return np.asarray(rows, dtype=np.int64), np.asarray(offsets, dtype=np.int64)
