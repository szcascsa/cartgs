"""完整结果后的人工判读固化；不改变checkpoint、uniform或selected sets。"""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent;path=ROOT/'audit_summary.json';s=json.loads(path.read_text())
assert s['correctness']['all_passed']
assert s['counts']['Grid-MLP']['Raw-Hard-L-Grid-MLP']['eval']['psnr']['improved']==0
assert s['counts']['Independent-Cell']['Raw-Cached']['eval']['psnr']['improved']==24
s['interpretation']=(
    '**结论：Cached推理没有修复Grid-MLP的PSNR/photo问题，LPIPS存在明显取舍。** '
    '相对同uniform初始化Raw-Cached，Grid-MLP在non-Bank仅10/24个PSNR改善，平均ΔPSNR=-0.135063 dB；'
    '相同实际L下比learned-score hard Top-L低0.518126 dB，比Raw hard Top-L低0.995056 dB。'
    'Independent-Cell相对Raw-Cached的PSNR/photo均24/24改善，PSNR平均+0.169661 dB，'
    '但仍低于same-L Raw hard的整体PSNR/photo。Cached相对hard的LPIPS多数更好，不能把四指标概括为一致变差。')
s['discussion']=[
    'Grid-MLP与原hard Top-K相比，non-Bank PSNR仅5/24改善、平均-0.549823 dB，photo为0/24改善；严格same-L比较learned hard Top-L仍平均-0.518126 dB，说明PSNR/photo损失不只是实际点数波动。',
    'Grid-MLP即使使用与训练相同的Cached推理，Bank相对初始化Raw-Cached也只有8/24个PSNR、9/24个photo改善；non-Bank分别10/24、11/24。这削弱了“只要把hard导出改成forward采样，MLP训练就会有效”的解释。',
    'Independent-Cell在相同Cached策略内出现一致学习信号：Bank/non-Bank的PSNR、SSIM、photo均24/24优于初始化；non-Bank LPIPS为17/24改善。这支持固定regional系数能够在当前Cached协议下学到信号，但23/24个case与初始化实际点数不同，不能把全部收益归因于更好的等预算选择；也没有测试更换uniform后的迁移。',
    '相同L的Raw hard比较中，Grid-MLP PSNR/photo均0/24改善，Independent-Cell PSNR仅1/24、photo0/24改善，平均ΔPSNR分别-0.995056/-0.731109 dB。因此Cached协议内的学习进展尚未转化为超过已有Raw hard的重建质量。',
    '四指标出现取舍：Grid-MLP Cached比same-L Raw hard的LPIPS在24/24个case更好，平均ΔLPIPS=-0.037025；同时PSNR/SSIM/photo全部更差。Independent-Cell的对应LPIPS为23/24更好。不能只看PSNR否认LPIPS收益，也不能只看LPIPS宣称整体方案成功。',
    '这次结果收窄了此前机制判断：训练sample与hard集合不一致是事实，但它不自动等价于hard导出在损害质量。实际把推理对齐forward后，PSNR/photo总体更差。当前Grid-MLP还存在参数共享/优化本身的问题，单纯换导出不是充分修复；本轮无法进一步区分MLP容量、平滑共享和优化条件。',
    '本轮保留Cached实验推理模式与全部对照，不把它部署到production，不自动改LR、network、grid、sampling或loss，也不重新训练。'
]
path.write_text(json.dumps(s,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');print('INTERPRETATION_FIXED')
