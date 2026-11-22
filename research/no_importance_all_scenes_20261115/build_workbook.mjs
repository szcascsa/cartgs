import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root = path.dirname(new URL(import.meta.url).pathname.replace(/^\/(\w:)/, '$1'));
const output = path.join(decodeURIComponent(root), 'outputs/01a1de13-ebef-7091-b328-52d3c81cfc40');
const data = JSON.parse(await fs.readFile(path.join(output, '分析数据.json'), 'utf8'));
const wb = Workbook.create();
const summary = wb.worksheets.add('汇总');
const detail = wb.worksheets.add('场景指标');
const status = wb.worksheets.add('场景状态');
const exposure = wb.worksheets.add('训练曝光');
const scope = { global: '全局', view_ratio: '单视角', full: 'Full' };
const mode = { mono: '单目', rgbd: 'RGB-D', stereo: '双目' };
const deltaFormat = '+0.000;-0.000;0.000';
const pctFormat = '+0.0%;-0.0%;0.0%';
const rangeEnd = 7 + data.detail.length;

function setup(sh, last, title, notes, headers, rows, tableName) {
  sh.showGridLines = false;
  sh.tabColor = '#234B68';
  sh.getRange('A1:M1').merge();
  sh.getRange('A1').values = [[title]];
  notes.forEach((s, i) => {
    sh.getRange(`A${i + 2}:M${i + 2}`).merge();
    sh.getRange(`A${i + 2}`).values = [[s]];
  });
  sh.getRange(`A7:${last}7`).values = [headers];
  sh.getRange(`A8:${last}${7 + rows.length}`).values = rows;
  sh.tables.add(`A7:${last}${7 + rows.length}`, true, tableName);
  const used = sh.getRange(`A1:${last}${7 + rows.length}`);
  used.format.font = { name: 'Microsoft YaHei', size: 10, color: '#253746' };
  used.format.rowHeight = 21;
  used.format.columnWidth = 14;
  used.format.verticalAlignment = 'center';
  sh.getRange(`A1:${last}1`).format = {
    fill: '#18364C', font: { name: 'Microsoft YaHei', size: 16, bold: true, color: '#FFFFFF' }, rowHeight: 34,
  };
  sh.getRange(`A2:${last}5`).format = { font: { size: 10, color: '#4B5563' }, rowHeight: 23 };
  sh.getRange(`A7:${last}7`).format = {
    fill: '#315C78', font: { bold: true, color: '#FFFFFF' }, wrapText: true, rowHeight: 40,
  };
  sh.freezePanes.freezeRows(7);
  sh.freezePanes.freezeColumns(3);
}

function deltaColors(sh, column, end, higherBetter = true) {
  const r = sh.getRange(`${column}8:${column}${end}`);
  r.setNumberFormat(deltaFormat);
  for (const [operator, good] of [['greaterThan', higherBetter], ['lessThan', !higherBetter]]) {
    r.conditionalFormats.add('cellIs', { operator, formula: 0,
      format: { fill: good ? '#E4F1EB' : '#FBE9E7', font: { color: good ? '#176047' : '#A53329' } } });
  }
}

const detailRows = data.detail.map(r => [r.dataset, r.scene, mode[r.mode], scope[r.scope], r.ratio,
  r.ratio_kind, r.status, r.gaussian_N, r.anchor_N, r.frames,
  r.psnr_A, r.psnr_D, null, r.ssim_A, r.ssim_D, null, r.lpips_A, r.lpips_D, null,
  r.actual_K_mean_A, r.actual_K_mean_D, r.actual_K_min_A, r.actual_K_min_D,
  r.actual_K_max_A, r.actual_K_max_D, r.ate_rmse_A, r.ate_rmse_D,
  r.render_ms_A, r.render_ms_D, r.psnr_paired_median ?? null,
  r.psnr_paired_p10 ?? null, r.psnr_paired_p90 ?? null, r.psnr_improved_frame_fraction ?? null]);
setup(detail, 'AG', '各场景、模式与ratio的完整对照', [
  'A：两网络真实视角＋GI=1、LI=0.1；D：相同网络，GI=0、LI=0。全部为seed0。',
  'Δ=D−A。失败组保留A参考值，D及差值为空白；所有K均为实际推理选点数。',
  'ATE为固定轨迹误差；渲染耗时来自不同运行环境，仅供记录，不能直接作为性能消融。',
  '来源：服务器汇总A_VS_D_METRICS、LOD_METRICS及PAIRED_FRAME_SUMMARY；对应CSV保留模型路径。',
], ['数据集', '场景', '模式', '范围', 'ratio', 'ratio类型', 'D状态', 'Gaussian N', 'Anchor N', '评价帧数',
  'A PSNR', 'D PSNR', 'ΔPSNR', 'A SSIM', 'D SSIM', 'ΔSSIM', 'A LPIPS', 'D LPIPS', 'ΔLPIPS',
  'A K均值', 'D K均值', 'A K最小', 'D K最小', 'A K最大', 'D K最大', 'A ATE(m)', 'D ATE(m)',
  'A 渲染(ms)', 'D 渲染(ms)', '逐帧ΔPSNR median', '逐帧ΔPSNR p10', '逐帧ΔPSNR p90', 'PSNR改善帧占比'], detailRows, 'SceneMetrics');
detail.getRange('B:B').format.columnWidth = 39;
detail.getRange('G:G').format.columnWidth = 23;
detail.getRange('A:A').format.columnWidth = 14;
detail.getRange(`E8:E${rangeEnd}`).setNumberFormat('0%');
detail.getRange(`H8:J${rangeEnd}`).setNumberFormat('#,##0');
detail.getRange(`K8:S${rangeEnd}`).setNumberFormat('0.0000');
detail.getRange(`K8:M${rangeEnd}`).setNumberFormat('0.000');
detail.getRange(`T8:Y${rangeEnd}`).setNumberFormat('#,##0.0');
detail.getRange(`Z8:AA${rangeEnd}`).setNumberFormat('0.000000');
detail.getRange(`AB8:AF${rangeEnd}`).setNumberFormat('0.000');
detail.getRange(`AG8:AG${rangeEnd}`).setNumberFormat('0.0%');
for (const [dst, a, d] of [['M', 'K', 'L'], ['P', 'N', 'O'], ['S', 'Q', 'R']]) {
  detail.getRange(`${dst}8`).formulas = [[`=IF(${d}8="","",${d}8-${a}8)`]];
  detail.getRange(`${dst}8:${dst}${rangeEnd}`).fillDown();
  deltaColors(detail, dst, rangeEnd, dst !== 'S');
}

const summaryRows = data.summary.map(r => [r.dataset, scope[r.scope], r.ratio, null,
  null, null, null, null, null, null, null, null, null, null]);
setup(summary, 'N', '去掉GI与LI后的渲染质量', [
  '32组：27组完成，5组连续零选点停止；已完成组共621项全轨迹评价。',
  '仅对27组共同成功场景／模式等权平均；失败组不记零分。PSNR、SSIM越高越好，LPIPS越低越好。',
  'A历史保留长尾／饱和停止与重试；D关闭长尾停止。该差异及单seed范围限制见报告。',
  '汇总公式读取“场景指标”。训练ratio：1%、5%、10%、30%、50%、70%；其余3%、8%、20%、40%、60%仅评价。',
], ['数据集', '范围', 'ratio', '成功组数', 'A PSNR', 'D PSNR', 'ΔPSNR', 'A SSIM', 'D SSIM', 'ΔSSIM',
  'A LPIPS', 'D LPIPS', 'ΔLPIPS', 'PSNR改善组数'], summaryRows, 'DatasetSummary');
summary.getRange('A:A').format.columnWidth = 16;
summary.getRange('C8:C99').setNumberFormat('0%');
summary.getRange('E8:G99').setNumberFormat('0.000');
summary.getRange('H8:M99').setNumberFormat('0.0000');
summary.getRange('D8:D99').setNumberFormat('0');
summary.getRange('N8:N99').setNumberFormat('0');
function dr(c) { return `'场景指标'!$${c}$8:$${c}$${rangeEnd}`; }
for (let i = 0; i < data.summary.length; ++i) {
  const row = i + 8;
  const criteria = [`${dr('G')},"完成"`, `${dr('D')},B${row}`, `${dr('E')},C${row}`];
  if (data.summary[i].dataset !== '全部') criteria.push(`${dr('A')},A${row}`);
  const filter = criteria.join(',');
  summary.getRange(`D${row}`).formulas = [[`=COUNTIFS(${filter})`]];
  for (const [dest, src] of [['E', 'K'], ['F', 'L'], ['H', 'N'], ['I', 'O'], ['K', 'Q'], ['L', 'R']]) {
    summary.getRange(`${dest}${row}`).formulas = [[`=SUMIFS(${dr(src)},${filter})/$D${row}`]];
  }
  for (const [dest, a, d] of [['G', 'E', 'F'], ['J', 'H', 'I'], ['M', 'K', 'L']]) {
    summary.getRange(`${dest}${row}`).formulas = [[`=${d}${row}-${a}${row}`]];
  }
  summary.getRange(`N${row}`).formulas = [[`=COUNTIFS(${filter},${dr('M')},">0")`]];
}
for (const c of ['G', 'J', 'M']) deltaColors(summary, c, 99, c !== 'M');

setup(status, 'Q', '32组实验状态与失败现场', [
  '完成27组；5组因连续零选点停止，没有将失败中间模型用于正式评价。',
  '已完成Fine步数来自real_lod_steps.csv；触发Fine步与累计Mapping步分列，避免混淆。',
  '全局连续零／同ratio连续零，任一达到8且存在合格候选时停止。Scale／XYZ长尾仅记录。',
  '来源：LOD_PIPELINE、各模型lod_health_failure和FAILED_STEP_AUDIT。',
], ['数据集', '场景', '模式', '状态', '目标Fine步', '完成Fine步', '失败触发Fine步', '累计Mapping步',
  '失败分支', '失败ratio', '合格候选', '触发K', '连续全局零', '同ratio连续零', 'Gaussian N', 'Anchor N', '完成评价数'],
data.status.map(r => [r.dataset, r.scene, mode[r.mode], r.status, r.target_steps, r.completed_steps,
  r.failure_trigger_fine_step, r.failure_step_raw, r.failure_error ? (r.failure_error.startsWith('view_budget') ? '单视角' : '全局') : null,
  r.failure_ratio, r.eligible_candidates, r.selected_at_failure, r.zero_streak_global, r.zero_streak_ratio,
  r.gaussian_N, r.anchor_N, r.evaluation_cases]), 'RunStatus');
status.getRange('B:B').format.columnWidth = 39;
status.getRange('D:D').format.columnWidth = 24;
status.getRange('H:H').format.columnWidth = 19;
status.getRange('E8:H39').setNumberFormat('#,##0');
status.getRange('J8:J39').setNumberFormat('0%');
status.getRange('K8:Q39').setNumberFormat('#,##0');

const exposureRows = data.exposure.map(r => {
  const [ds, scene, m] = r.id.split('/');
  return [ds, scene, mode[m], r.steps, r.reference_global_K, r.selection_exposure_global, null,
    r.reference_local_K, r.selection_exposure_local, null, r.importance_renders, r.li_capture, r.li_vjp,
    r.feature_adam_clocks.join(', '), r.nominal_feature_rotation_clocks.join(', '),
    r.both_branches_zero_steps.join(', '), r.feature_adam_clocks_match_gradient_exposure ? '一致' : '不一致'];
});
setup(exposure, 'Q', '实际选点曝光与Anchor Feature更新', [
  '仅包含27组成功运行。A与D使用相同相机／ratio序列、Fine步数，但实际训练K曝光可以不同。',
  'GI采集刷新仍运行，D不使用GI Loss；D的LI capture与LI VJP均为0。',
  'room2单目Fine第12703步两分支K均为0，Base Feature无梯度，Adam少1次更新符合原语义。',
  '来源：TRAINING_EXPOSURE；Adam时钟按Base、Residual1至5顺序记录。',
], ['数据集', '场景', '模式', 'Fine步数', 'A全局累计K', 'D全局累计K', '全局曝光变化', 'A局部累计K', 'D局部累计K',
  '局部曝光变化', 'GI采集渲染', 'LI采集次数', 'LI VJP次数', '实际Feature Adam时钟', '名义ratio时钟', '双分支K=0步骤', '梯度曝光核验'],
exposureRows, 'TrainingExposure');
exposure.getRange('B:B').format.columnWidth = 39;
exposure.getRange('N:O').format.columnWidth = 49;
exposure.getRange('P:Q').format.columnWidth = 22;
exposure.getRange('D8:M34').setNumberFormat('#,##0');
for (const [dst, a, d] of [['G', 'E', 'F'], ['J', 'H', 'I']]) {
  exposure.getRange(`${dst}8`).formulas = [[`=${d}8/${a}8-1`]];
  exposure.getRange(`${dst}8:${dst}34`).fillDown();
  exposure.getRange(`${dst}8:${dst}34`).setNumberFormat(pctFormat);
}

// 验证公式重算，以及缺失D仍为空白。
wb.recalculate();
const before = detail.getRange('L8').values[0][0];
detail.getRange('L8').values = [[before + 1]];
wb.recalculate();
assert.ok(Math.abs(summary.getRange('G8').values[0][0] - data.summary[0].delta_psnr - 1 / 27) < 1e-10,
  JSON.stringify({ before, perturbed: summary.getRange('D8:N8').values, formulas: summary.getRange('D8:N8').formulas,
    detail: detail.getRange('K8:M8').values, expected: data.summary[0].delta_psnr + 1 / 27 }));
detail.getRange('L8').values = [[before]];
wb.recalculate();
const missing = data.detail.filter(r => r.psnr_D === null).length;
assert.equal(missing, 115);
for (let i = 0; i < data.summary.length; ++i) {
  const actual = summary.getRange(`D${i + 8}:N${i + 8}`).values[0];
  const r = data.summary[i];
  const expected = [r.count, r.psnr_A, r.psnr_D, r.delta_psnr, r.ssim_A, r.ssim_D, r.delta_ssim,
    r.lpips_A, r.lpips_D, r.delta_lpips, r.psnr_improved_scenes];
  expected.forEach((v, j) => assert.ok(typeof actual[j] === 'number' && Math.abs(actual[j] - v) < 1e-10,
    `summary row ${i + 8} col ${j + 4}: ${actual[j]} != ${v}`));
}
for (let i = 0; i < data.detail.length; ++i) {
  if (data.detail[i].psnr_D !== null) continue;
  const v = detail.getRange(`M${i + 8}`).values[0][0];
  assert.ok(v === '' || v === null, `missing D converted to ${v}`);
}
const errorScan = await wb.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options: { useRegex: true, maxResults: 50 }, summary: '最终公式错误检查', maxChars: 3000 });
const inspect = await wb.inspect({ kind: 'table', range: '汇总!A7:N13', include: 'values,formulas', tableMaxRows: 7, tableMaxCols: 14, maxChars: 5000 });
const previewDir = path.join(decodeURIComponent(root), 'workbook_previews');
await fs.mkdir(previewDir, { recursive: true });
for (const [sheetName, range] of [['场景状态', 'A1:Q14'], ['训练曝光', 'A1:Q14']]) {
  const png = await wb.render({ sheetName, range, scale: 1.4, format: 'png' });
  await fs.writeFile(path.join(previewDir, `${sheetName}.png`), new Uint8Array(await png.arrayBuffer()));
}
const file = path.join(output, '无重要性消融对照.xlsx');
await (await SpreadsheetFile.exportXlsx(wb)).save(file);
await fs.writeFile(path.join(previewDir, 'verification.json'), JSON.stringify({ summaryRows: 92, sceneRows: 736,
  matchedEvaluations: 621, failedBlankRows: missing, formulaPerturbation: '通过', errorScan: errorScan.ndjson, inspect: inspect.ndjson }, null, 2));
console.log(JSON.stringify({ file, summaryRows: 92, sceneRows: 736, failedBlankRows: missing, errorScan: errorScan.ndjson }));
