'use strict';
const $ = id => document.getElementById(id);
const fmt = value => value == null ? '—' : Number(value).toLocaleString('zh-CN');
const fixed = (value, digits = 2) => value == null ? '—' : typeof value === 'string' ? value : Number(value).toFixed(digits);
const percent = value => value == null ? '—' : fixed(100 * value) + '%';
const db = value => value == null ? '—' : fixed(value) + ' dB';
const copy = pose => pose.map(row => row.slice());
const modeNames = {budget: '可见预算', global_ratio: '全图训练 ratio', full: 'Full Master', ranked_lod: 'Anchor 分档'};
const state = {frame: 0, free: false, pose: null, mode: 'budget', budget: 5000, unit: 'points', ratio: .4,
  trainingRatio: .2, ablation: 0, fullReference: false, anchorLevels: [0, 1, 2], imageFormat: 'jpeg',
  viewConditioned: false, lodMixed: false, lodSorting: 'projection', lodRatios: [.5, .3, .1],
  lodShares: [.2, .3, .5], lodHysteresis: false, lodColorize: false, lodEquivalentRatio: .3,
  lodMixedSeed: 0, lodMixedSample: 0};
let meta, revision = 0, completedRevision = -1, busy = false, scheduled = null;
let playing = false, playTimer = null, lastDisplayed = null, referenceCacheKey = null;
let resetEpoch = 0, resetSent = 0, drag = null, animation = null, previousTick = 0;
const heldKeys = new Set();

function error(message) { $('errorText').textContent = message; $('error').hidden = false; }
function status(message, active = false) { $('status').textContent = message; $('status').classList.toggle('busy', active); }
function invalidate() {
  if (!meta) return;
  revision++;
  clearTimeout(playTimer);
  $('displayPending').hidden = false;
  status('更新中', true);
  // 连续输入不重置定时器，防止一直拖动/按键时请求永远无法发送。
  if (!busy && scheduled == null) scheduled = setTimeout(pump, 25);
}
function renderSettings() {
  const request = {mode: state.mode, budget: state.budget, budget_unit: state.unit,
    ratio: state.mode === 'global_ratio' ? state.trainingRatio : state.ratio,
    selection: state.ablation < 2 ? 'learned' : 'random', compensation: state.ablation % 2 === 0,
    anchor_levels: state.anchorLevels.slice(), reference_full: state.fullReference,
    reference_cache_key: referenceCacheKey, image_format: state.imageFormat,
    view_conditioned: state.viewConditioned && ['budget', 'ranked_lod'].includes(state.mode)};
  if (state.free) request.pose = copy(state.pose); else request.frame = state.frame;
  if (state.mode === 'ranked_lod') Object.assign(request, {
    lod_sorting: state.lodMixed ? 'projection' : state.lodSorting, lod_mixed: state.lodMixed,
    lod_equivalent_ratio: state.lodEquivalentRatio, lod_mixed_seed: state.lodMixedSeed,
    lod_mixed_sample: state.lodMixedSample, lod_ratios: state.lodRatios.slice(), lod_shares: state.lodShares.slice(),
    lod_hysteresis: !state.lodMixed && state.lodHysteresis ? .02 : 0,
    lod_colorize: state.lodColorize, lod_reset_history: resetEpoch > resetSent});
  return request;
}
async function decodeImage(source) {
  const image = new Image(); image.src = source;
  await image.decode();
  return image;
}
async function pump() {
  scheduled = null;
  if (busy || !meta) return;
  busy = true;
  const version = revision, request = renderSettings(), resetVersion = resetEpoch, started = performance.now();
  request.request_id = version;
  const abort = new AbortController(), timeout = setTimeout(() => abort.abort(), 120000);
  try {
    const response = await fetch('/api/render', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(request), signal: abort.signal});
    const text = await response.text(), roundtrip = performance.now() - started;
    const data = JSON.parse(text);
    if (!response.ok) throw Error(data.error || response.statusText);
    const decodeStarted = performance.now();
    await Promise.all([decodeImage(data.image), data.reference ? decodeImage(data.reference) : Promise.resolve()]);
    const timing = {roundtrip_ms: roundtrip, decode_ms: performance.now() - decodeStarted,
      display_ms: performance.now() - started, response_bytes: Number(response.headers.get('Content-Length')) || new TextEncoder().encode(text).length};
    // 单个在途请求按完成顺序显示；持续移动时仍显示已完成帧，不等到松键才刷新。
    show(data, request, timing);
    completedRevision = version;
    if (request.lod_reset_history) resetSent = resetVersion;
    $('error').hidden = true;
  } catch (failure) {
    error(failure.name === 'AbortError' ? '渲染请求超过120秒。可使用刷新重试；不会自动堆积请求。' : failure.message);
    status('渲染失败'); stopPlay();
  } finally {
    clearTimeout(timeout); busy = false;
    if (revision !== version) {
      scheduled = setTimeout(pump, 0);
    } else {
      $('displayPending').hidden = true;
      if (completedRevision === version) status('已更新');
      if (playing) schedulePlayback(Math.max(0, 1000 / Number($('playRate').value) - (performance.now() - started)));
    }
  }
}
function setText(id, value) { $(id).textContent = value; }
function updateLevels(levels = []) {
  const rows = levels.map(level => {
    setText('anchorLevel' + level.level + 'Count', fmt(level.anchors) + ' / ' + fmt(level.gaussians) + ' GS');
    const row = document.createElement('tr');
    if (level.enabled === false) row.className = 'disabled';
    for (const value of [level.level, fmt(level.anchors), fmt(level.gaussians), fmt(level.visible_gaussians), fmt(level.submitted_gaussians)]) {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    }
    return row;
  });
  $('anchorRows').replaceChildren(...rows);
}
function updateGroups(s) {
  const rows = (s.lod_groups || []).map(group => {
    const row = document.createElement('tr');
    for (const value of [fixed(group.ratio, 4), percent(group.actual_share), fmt(group.anchors), fmt(group.candidate_gaussians), fmt(group.active_gaussians)]) {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    }
    return row;
  });
  $('lodRows').replaceChildren(...rows);
  setText('groupSummary', s.mode !== 'ranked_lod' ? 'Anchor 分档模式下显示。' :
    (s.mixed_ratio ? '目标 ' + percent(s.target_equivalent_ratio) + ' · sample ' + s.mixed_sample + ' · ' : '') +
    '加权 ratio ' + percent(s.weighted_target_ratio) + ' · 实际保留 ' + percent(s.actual_retention_ratio));
}
function show(data, request, timing) {
  const s = data.stats;
  $('renderImage').src = data.image; $('renderImage').hidden = false; $('renderEmpty').hidden = true;
  if (!data.reference_available) {
    $('referenceImage').hidden = true; $('referenceEmpty').hidden = false;
    $('referenceImage').removeAttribute('src'); referenceCacheKey = null;
  } else if (data.reference) {
    $('referenceImage').src = data.reference; $('referenceImage').hidden = false; $('referenceEmpty').hidden = true;
    referenceCacheKey = data.reference_cache_key;
  } else if (!data.reference_unchanged || referenceCacheKey !== data.reference_cache_key) {
    // 只有确实在浏览器保留了对应参考图，才向服务器声明缓存命中。
    referenceCacheKey = null; $('referenceImage').hidden = true; $('referenceEmpty').hidden = false;
  }
  setText('referenceLabel', data.reference_label);
  setText('viewKind', s.view_kind === 'dataset' ? '数据集相机' : '自由相机');
  setText('viewTitle', s.view_kind === 'dataset' ? meta.frames[s.frame].name : '自由观察 · 无匹配RGB');
  setText('actualK', fmt(s.active_gaussians));
  setText('budgetDetail', s.mode === 'full' ? '所选层级的原始 Master' : '请求 ' + fmt(s.requested_budget) + ' · 有效 ' + fmt(s.effective_budget));
  setText('psnr', db(s.ground_truth_psnr));
  setText('psnrDetail', s.ground_truth_psnr == null ? '当前视角无匹配RGB真值' : '编码前 FP32 渲染与RGB比较');
  setText('fullPsnr', s.full_reference_psnr == null ? '未开启' : db(s.full_reference_psnr));
  setText('latency', fixed(s.prepare_plus_render_ms) + ' ms');
  setText('imageNote', modeNames[s.mode] + ' · ' + s.ablation_group + ' · ' + fmt(s.active_gaussians) + ' GS · ' + meta.calibration.width + ' × ' + meta.calibration.height);
  setText('selectionScope', {global: '全图', visible_anchors: '视锥可见 Anchor', ranked_visible_anchors: '可见 Anchor 分组', full_master: '原始 Master'}[s.selection_scope] || s.selection_scope);
  setText('ablationValue', s.ablation_group + (s.selection === 'full' ? '' : ' · ' + (s.selection === 'learned' ? '学习选择' : '随机选择')));
  setText('ratioValue', s.mode === 'ranked_lod' ? '按空间分组' : percent(s.ratio));
  setText('retentionValue', s.candidate_gaussians ? percent(s.active_gaussians / s.candidate_gaussians) : '—');
  const numbers = {candidateN: 'candidate_gaussians', eligibleN: 'eligible_candidates', shortfall: 'budget_shortfall', projectedN: 'projected_gaussians'};
  for (const [id, key] of Object.entries(numbers)) setText(id, fmt(s[key]));
  setText('budgetValues', fmt(s.requested_budget) + ' / ' + fmt(s.effective_budget));
  setText('enabledAnchors', fmt(s.enabled_anchors) + ' / ' + fmt(s.enabled_anchor_gaussians));
  setText('enabledViewAnchors', fmt(s.enabled_visible_anchors) + ' / ' + fmt(s.enabled_visible_gaussians));
  setText('submittedAnchors', fmt(s.submitted_anchors) + ' / ' + fmt(s.projected_anchors));
  $('pointBar').style.width = Math.min(100, 100 * s.active_gaussians / Math.max(1, s.total_gaussians)) + '%';
  $('ratioWarning').hidden = s.ratio_in_training_range !== false;
  $('budgetWarning').hidden = !s.budget_shortfall;
  setText('budgetWarning', '可选点不足，实际 K 比请求少 ' + fmt(s.budget_shortfall) + '；未使用隐藏层级的点补齐。');
  updateLevels(s.anchor_levels); updateGroups(s);
  setText('cameraPosition', 'X ' + fixed(data.pose[0][3], 4) + '  Y ' + fixed(data.pose[1][3], 4) + '  Z ' + fixed(data.pose[2][3], 4));
  setText('cameraMatch', s.view_kind === 'dataset' ? '已显示第 ' + (s.frame + 1) + ' 帧 · RGB精确匹配时计算PSNR' : '自由相机 · 不使用最近RGB冒充真值');
  const msFields = {selectorMs: 'selector_ms', compensationMs: 'compensation_ms', prepareMs: 'prepare_ms', rasterMs: 'rasterizer_ms',
    summaryMs: 'anchor_summary_ms', qualityMs: 'quality_and_reference_ms', referenceMs: 'reference_render_ms', colorMs: 'lod_debug_render_ms', serverMs: 'server_request_ms'};
  for (const [id, key] of Object.entries(msFields)) setText(id, fixed(s[key] == null && id === 'colorMs' ? 0 : s[key]));
  setText('groupingMs', fixed(s.grouping_ms || 0) + ' / ' + fixed(s.budget_calibration_ms || 0));
  setText('encodeMs', fixed(s.image_encode_ms) + ' / ' + fixed(s.reference_encode_ms));
  setText('roundtripMs', fixed(timing.roundtrip_ms)); setText('decodeMs', fixed(timing.decode_ms));
  setText('displayMs', fixed(timing.display_ms)); setText('responseBytes', fixed(timing.response_bytes / 1024, 1) + ' KiB');
  setText('backendValue', s.inference_backend || meta.inference_backend);
  setText('cacheValue', s.mode === 'full' ? 'Full数据已驻留' : s.cache_hit ? '命中' : '本次计算');
  setText('referenceCacheValue', data.reference_unchanged ? '命中 · 未重复发送' : data.reference ? '已发送' : '无参考图');
  setText('graphValue', fmt(s.graph_count) + ' / ' + fixed(s.graph_cache_mb, 1) + ' MiB');
  setText('memoryValue', fixed(s.gpu_allocated_mb, 0) + ' / ' + fixed(s.gpu_reserved_mb, 0) + ' MiB');
  const parts = [['准备', s.prepare_ms, '#65dfbf'], ['光栅化', s.rasterizer_ms, '#78aff5'], ['统计', s.anchor_summary_ms, '#ba9af2'],
    ['质量与参考', s.quality_and_reference_ms, '#eabe70'], ['编码', (s.image_encode_ms || 0) + (s.reference_encode_ms || 0), '#dc8eb3']];
  const total = parts.reduce((sum, p) => sum + (p[1] || 0), 0);
  $('timingBars').replaceChildren(...parts.filter(p => p[1] > 0).map(([label, value, color]) => {
    const bar = document.createElement('span'); bar.style.width = (100 * value / total) + '%';
    bar.style.background = color; bar.title = label + ' ' + fixed(value) + ' ms'; return bar;
  }));
  setText('browserTiming', '往返 ' + fixed(timing.roundtrip_ms) + ' · 解码 ' + fixed(timing.decode_ms) + ' · 合计 ' + fixed(timing.display_ms) + ' ms');
  setText('activityText', revision === request.request_id ? '已显示当前设置 · 空闲时不发送请求' : '已显示最近完成帧 · 正在处理最新操作');
  lastDisplayed = {request: {...request, reference_cache_key: undefined}, pose: data.pose, stats: s, browser: timing, image: data.image};
  $('saveImage').disabled = $('saveMetrics').disabled = false;
  window.viewerLastStats = s;
}

function syncControls() {
  if (!meta) return;
  document.querySelectorAll('[data-mode]').forEach(button => button.setAttribute('aria-pressed', button.dataset.mode === state.mode));
  document.querySelectorAll('[data-unit]').forEach(button => button.setAttribute('aria-pressed', button.dataset.unit === state.unit));
  document.querySelectorAll('[data-ablation]').forEach(button => button.setAttribute('aria-pressed', Number(button.dataset.ablation) === state.ablation));
  $('budgetControls').hidden = state.mode !== 'budget';
  $('pointControls').hidden = state.unit !== 'points'; $('ratioControls').hidden = state.unit !== 'ratio';
  $('globalControls').hidden = state.mode !== 'global_ratio'; $('fullHelp').hidden = state.mode !== 'full';
  $('rankedControls').hidden = state.mode !== 'ranked_lod'; $('ablationControls').hidden = state.mode === 'full';
  $('lodMixedControls').hidden = !state.lodMixed; $('lodFixedControls').hidden = state.lodMixed;
  $('viewConditionControls').hidden = !meta.view_residual_available || !['budget', 'ranked_lod'].includes(state.mode) || state.ablation >= 2;
  setText('modeHelp', {budget: '仅从当前视锥内的 Anchor 成员中选择。', global_ratio: '与训练一致：全图选点，不使用当前视锥。',
    full: '原始 Master 参考。', ranked_lod: '先筛选可见 Anchor，再按 Gaussian 数量累计分组。'}[state.mode]);
  setText('mixedSample', state.lodMixedSample);
}
function numberInput(id, minimum, maximum, update, integer = false) {
  const input = $(id);
  input.addEventListener('input', () => {
    const value = input.valueAsNumber;
    const valid = Number.isFinite(value) && value >= minimum && value <= maximum && (!integer || Number.isSafeInteger(value));
    input.setCustomValidity(valid ? '' : '请输入范围内的有效数字'); input.setAttribute('aria-invalid', !valid);
    if (valid) update(value);
  });
}
function budget(value) {
  state.budget = value; $('budget').value = value;
  $('budgetRange').max = Math.max(meta.total_gaussians, value, 1); $('budgetRange').value = value; invalidate();
}
function ratio(value) { state.ratio = value; $('ratioBudget').value = value; $('ratioRange').value = value; invalidate(); }
function stopPlay() { playing = false; clearTimeout(playTimer); $('play').setAttribute('aria-pressed', 'false'); setText('play', '播放'); }
function frame(index, autoplay = false) {
  if (!meta) return;
  if (!autoplay) stopPlay();
  clearMotion();
  state.frame = (Math.round(index) % meta.frames.length + meta.frames.length) % meta.frames.length;
  state.free = false; state.pose = copy(meta.frames[state.frame].pose);
  $('timeline').value = $('frameSelect').value = state.frame; $('frameNumber').value = state.frame + 1;
  setText('frameCounter', (state.frame + 1) + ' / ' + meta.frames.length); invalidate();
}
function schedulePlayback(delay = 0) {
  clearTimeout(playTimer);
  if (!playing || document.hidden) return;
  playTimer = setTimeout(() => {
    if (!playing || busy) return;
    if (state.frame >= meta.frames.length - 1) return stopPlay();
    frame(state.frame + 1, true);
  }, delay);
}
function freeCamera() {
  stopPlay();
  if (!state.free) { state.pose = copy(meta.frames[state.frame].pose); state.free = true; }
}
function moveCamera(x, y, z, pitch = 0, yaw = 0, roll = 0) {
  if (!meta || !(x || y || z || pitch || yaw || roll)) return;
  freeCamera();
  const p = state.pose, cx = Math.cos(pitch), sx = Math.sin(pitch), cy = Math.cos(yaw), sy = Math.sin(yaw), cz = Math.cos(roll), sz = Math.sin(roll);
  const r = [[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx], [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], [-sy, cy*sx, cy*cx]];
  const next = copy(p);
  for (let i = 0; i < 3; i++) {
    next[i][3] += p[i][0]*x + p[i][1]*y + p[i][2]*z;
    for (let j = 0; j < 3; j++) next[i][j] = p[i][0]*r[0][j] + p[i][1]*r[1][j] + p[i][2]*r[2][j];
  }
  state.pose = next; invalidate();
}
function controlValue(id) {
  const input = $(id), value = input.valueAsNumber;
  return Number.isFinite(value) && value >= Number(input.min) && value <= Number(input.max) ? value : Number(input.defaultValue);
}
function clearMotion() {
  heldKeys.clear(); drag = null;
  if (animation != null) cancelAnimationFrame(animation);
  animation = null; previousTick = 0;
}
function tick(now) {
  if (!heldKeys.size || document.activeElement !== $('view')) { clearMotion(); return; }
  const dt = Math.min(.05, Math.max(0, (now - previousTick) / 1000)); previousTick = now;
  const down = key => heldKeys.has(key) ? 1 : 0, fast = heldKeys.has('shift') ? 3 : 1;
  const v = controlValue('moveSpeed') * dt * fast, a = controlValue('rotationSpeed') * dt * fast;
  moveCamera((down('d')-down('a'))*v, (down('e')-down('q'))*v, (down('w')-down('s'))*v,
    (down('i')-down('k'))*a, (down('l')-down('j'))*a, (down('o')-down('u'))*a);
  animation = requestAnimationFrame(tick);
}
async function fullscreen() {
  try {
    if (document.fullscreenElement) await document.exitFullscreen(); else await $('stage').requestFullscreen();
  } catch (failure) { error('浏览器无法进入全屏：' + failure.message); }
}
function download(href, name) {
  const link = document.createElement('a'); link.href = href; link.download = name;
  document.body.append(link); link.click(); link.remove();
}

document.querySelectorAll('[data-mode]').forEach(button => button.onclick = () => { state.mode = button.dataset.mode; syncControls(); invalidate(); });
document.querySelectorAll('[data-unit]').forEach(button => button.onclick = () => { state.unit = button.dataset.unit; syncControls(); invalidate(); });
document.querySelectorAll('[data-ablation]').forEach(button => button.onclick = () => { state.ablation = Number(button.dataset.ablation); syncControls(); invalidate(); });
document.querySelectorAll('[data-budget]').forEach(button => button.onclick = () => { if (meta) budget(Number(button.dataset.budget)); });
document.querySelectorAll('[data-tab]').forEach(button => button.onclick = () => {
  document.querySelectorAll('[data-tab]').forEach(other => other.setAttribute('aria-pressed', other === button));
  for (const name of ['overview', 'performance', 'groups']) $(name + 'Panel').hidden = name !== button.dataset.tab;
});
numberInput('budget', 0, Number.MAX_SAFE_INTEGER, value => { if (meta) budget(value); }, true);
$('budgetRange').oninput = event => { if (meta) budget(Number(event.target.value)); };
$('allBudget').onclick = () => { if (meta) budget(meta.total_gaussians); };
numberInput('ratioBudget', 0, 1, ratio); $('ratioRange').oninput = event => ratio(Number(event.target.value));
$('trainingRatio').onchange = event => { state.trainingRatio = Number(event.target.value); invalidate(); };
$('fullReference').onchange = event => { state.fullReference = event.target.checked; invalidate(); };
$('viewConditioned').onchange = event => { state.viewConditioned = event.target.checked; invalidate(); };
for (let level = 0; level < 3; level++) $('anchorLevel' + level).onchange = () => {
  state.anchorLevels = [0, 1, 2].filter(i => $('anchorLevel' + i).checked); invalidate();
};
$('lodMixed').onchange = event => { state.lodMixed = event.target.checked; syncControls(); invalidate(); };
$('lodSorting').onchange = event => { state.lodSorting = event.target.value; invalidate(); };
numberInput('lodEquivalentRatio', 0, 1, value => {
  if (!meta) return;
  if (value < Math.min(...meta.trained_ratios) || value > Math.max(...meta.trained_ratios)) {
    $('lodEquivalentRatio').setCustomValidity('等效ratio必须在训练端点之间'); return;
  }
  state.lodEquivalentRatio = value; invalidate();
});
numberInput('lodMixedSeed', 0, Number.MAX_SAFE_INTEGER, value => { state.lodMixedSeed = value; invalidate(); }, true);
$('lodResample').onclick = () => { state.lodMixedSample++; syncControls(); invalidate(); };
const ratioIds = ['lodHighRatio', 'lodMiddleRatio', 'lodLowRatio'];
for (const id of ratioIds) numberInput(id, .001, 1, () => {
  const values = ratioIds.map(name => $(name).valueAsNumber), valid = values.every(Number.isFinite) && values[0] > values[1] && values[1] > values[2] && values[2] > 0 && values[0] <= 1;
  for (const name of ratioIds) $(name).setCustomValidity(valid ? '' : '必须满足高档 > 中档 > 低档');
  if (valid) { state.lodRatios = values; invalidate(); }
});
for (const id of ['lodHighShare', 'lodMiddleShare']) numberInput(id, 0, 1, () => {
  const high = $('lodHighShare').valueAsNumber, middle = $('lodMiddleShare').valueAsNumber;
  const valid = Number.isFinite(high + middle) && high >= 0 && middle >= 0 && high + middle <= 1;
  for (const name of ['lodHighShare', 'lodMiddleShare']) $(name).setCustomValidity(valid ? '' : '两项占比合计不能超过1');
  if (valid) { state.lodShares = [high, middle, Math.max(0, 1-high-middle)]; setText('lodLowShare', '低档 ' + percent(state.lodShares[2]) + '；按Gaussian数量累计，整Anchor不拆分。'); invalidate(); }
});
$('lodHysteresis').onchange = event => { state.lodHysteresis = event.target.checked; invalidate(); };
$('lodColorize').onchange = event => { state.lodColorize = event.target.checked; invalidate(); };
$('lodResetHistory').onclick = () => { resetEpoch++; invalidate(); };
$('imageFormat').onchange = event => { state.imageFormat = event.target.value; invalidate(); };
$('profileBrowser').onchange = event => { $('browserTiming').hidden = !event.target.checked; };
$('displayScale').onchange = event => { $('renderGrid').classList.toggle('native-pixels', event.target.value === 'native'); };
$('compare').onclick = () => {
  const enabled = $('compare').getAttribute('aria-pressed') !== 'true';
  $('compare').setAttribute('aria-pressed', enabled); $('referencePane').hidden = !enabled; $('renderGrid').classList.toggle('compare', enabled);
};
$('fullscreen').onclick = fullscreen;
document.addEventListener('fullscreenchange', () => { setText('fullscreen', document.fullscreenElement ? '退出全屏' : '全屏'); });
$('frameSelect').onchange = event => frame(Number(event.target.value));
$('timeline').oninput = event => frame(Number(event.target.value));
numberInput('frameNumber', 1, Number.MAX_SAFE_INTEGER, value => { if (meta && value <= meta.frames.length) frame(value - 1); }, true);
$('previous').onclick = () => frame(state.frame - 1); $('next').onclick = () => frame(state.frame + 1);
$('reset').onclick = () => frame(state.frame);
$('play').onclick = () => {
  if (!meta) return;
  if (playing) return stopPlay();
  playing = true; $('play').setAttribute('aria-pressed', 'true'); setText('play', '暂停');
  if (state.frame === meta.frames.length - 1) frame(0, true); else if (!busy) schedulePlayback();
};
$('playRate').onchange = () => { if (playing && !busy) schedulePlayback(); };
$('refresh').onclick = () => { if (!meta) init(); else invalidate(); };
$('dismissError').onclick = () => { $('error').hidden = true; };
$('saveImage').onclick = () => { if (lastDisplayed) download(lastDisplayed.image, 'realLOD_' + lastDisplayed.stats.mode + '_' + (lastDisplayed.stats.frame ?? 'free') + (lastDisplayed.image.startsWith('data:image/png') ? '.png' : '.jpg')); };
$('saveMetrics').onclick = () => {
  if (!lastDisplayed) return;
  const {image, ...record} = lastDisplayed;
  const blob = new Blob([JSON.stringify({checkpoint: meta.checkpoint, exported_at: new Date().toISOString(), ...record}, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob); download(url, 'realLOD_metrics.json'); setTimeout(() => URL.revokeObjectURL(url), 1000);
};

$('view').oncontextmenu = event => event.preventDefault();
$('view').onpointerdown = event => {
  if (!meta || event.button > 2) return;
  event.preventDefault(); stopPlay(); $('view').focus({preventScroll: true});
  drag = {x: event.clientX, y: event.clientY, button: event.button, pointer: event.pointerId};
  $('view').setPointerCapture(event.pointerId);
};
$('view').onpointermove = event => {
  if (!drag || event.pointerId !== drag.pointer) return;
  const dx = event.clientX - drag.x, dy = event.clientY - drag.y; drag.x = event.clientX; drag.y = event.clientY;
  const w = $('view').clientWidth, h = $('view').clientHeight;
  if (drag.button === 0) moveCamera(0, 0, 0, -Math.PI*dy/h*controlValue('mouseRotate'), Math.PI*dx/w*controlValue('mouseRotate'));
  else if (drag.button === 2) moveCamera(0, 0, 0, 0, 0, Math.PI*dx/w*controlValue('mouseRoll'));
  else moveCamera(dx/w*controlValue('mousePan'), -dy/h*controlValue('mousePan'), 0);
};
$('view').onpointerup = $('view').onpointercancel = $('view').onlostpointercapture = () => { drag = null; };
$('view').addEventListener('wheel', event => {
  if (!meta) return;
  event.preventDefault(); moveCamera(0, 0, -Math.sign(event.deltaY)*controlValue('moveSpeed')*.3);
}, {passive: false});
$('view').onkeydown = event => {
  const key = event.key.toLowerCase();
  if (!meta || event.ctrlKey || event.altKey || event.metaKey) return;
  if (('wasdqeikjluo'.includes(key) && key.length === 1) || key === 'shift') {
    event.preventDefault(); heldKeys.add(key);
    if (animation == null) { previousTick = performance.now(); animation = requestAnimationFrame(tick); }
  } else if (key === 'r' && !event.repeat) { event.preventDefault(); frame(state.frame); }
  else if (key === 'f' && !event.repeat) { event.preventDefault(); fullscreen(); }
};
window.addEventListener('keyup', event => { heldKeys.delete(event.key.toLowerCase()); });
$('view').addEventListener('blur', clearMotion); window.addEventListener('blur', () => { clearMotion(); stopPlay(); });
document.addEventListener('visibilitychange', () => { if (document.hidden) { clearMotion(); stopPlay(); } });

async function init() {
  try {
    status('正在连接', true);
    const response = await fetch('/api/meta');
    if (!response.ok) throw Error('模型服务连接失败，HTTP ' + response.status);
    const metadata = await response.json();
    if (!metadata.frames?.length || !metadata.trained_ratios?.length) throw Error('模型没有可用相机或训练ratio');
    meta = metadata;
    state.budget = meta.default_budget; state.trainingRatio = meta.trained_ratios[0];
    state.lodMixedSeed = meta.mixed_ratio_defaults?.seed || 0;
    state.lodEquivalentRatio = Math.min(Math.max(.3, Math.min(...meta.trained_ratios)), Math.max(...meta.trained_ratios));
    state.viewConditioned = !!meta.view_residual_available;
    $('viewConditioned').checked = state.viewConditioned;
    setText('viewConditionLabel', meta.view_conditioning_type === 'residual' ? '视角残差 Selector' : '视角条件 Selector');
    $('budget').value = $('budgetRange').value = state.budget;
    $('budgetRange').max = Math.max(meta.total_gaussians, state.budget, 1);
    $('lodMixedSeed').value = state.lodMixedSeed; $('lodEquivalentRatio').value = state.lodEquivalentRatio;
    $('lodEquivalentRatio').min = Math.min(...meta.trained_ratios); $('lodEquivalentRatio').max = Math.max(...meta.trained_ratios);
    $('trainingRatio').replaceChildren(...meta.trained_ratios.map(value => new Option(percent(value), value)));
    $('frameSelect').replaceChildren(...meta.frames.map((item, index) => new Option((index + 1) + ' / ' + meta.frames.length + ' · ' + item.name, index)));
    $('timeline').max = meta.frames.length - 1; $('frameNumber').max = meta.frames.length;
    setText('totalLabel', fmt(meta.total_gaussians) + ' Gaussian · ' + fmt(meta.total_anchors) + ' Anchor');
    setText('backendBadge', {cuda: '融合 CUDA · 已启用', cuda_graph: 'CUDA Graph · 已启用', pytorch: 'PyTorch'}[meta.inference_backend] || '推理后端');
    $('backendBadge').classList.toggle('fallback', !!meta.inference_backend_note);
    $('backendBadge').title = meta.inference_backend_note || ('实际推理后端：' + meta.inference_backend);
    setText('backendNote', meta.inference_backend_note ? '此checkpoint使用PyTorch兼容路径：' + meta.inference_backend_note :
      '实际推理后端：' + meta.inference_backend + '。训练区间 ' + percent(Math.min(...meta.trained_ratios)) + '–' + percent(Math.max(...meta.trained_ratios)) + '。');
    setText('gpuName', meta.gpu_name || ''); setText('modelPath', meta.checkpoint);
    updateLevels(meta.anchor_levels); syncControls(); frame(0);
  } catch (failure) { error(failure.message); status('连接失败'); setText('backendBadge', '服务未连接'); }
}
init();
