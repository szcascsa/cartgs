# 真实数据清单



|图|数据文件|数据性质|来源|脚本|输出|

|---|---|---|---|---|---|

|8场景Mono/RGB-D Full比较|data/FULL_VS_VANILLA.json|真实实验|本次16个Full+tail checkpoint及48个Vanilla候选的已存全轨迹结果|plot_full_vs_vanilla.py（ZIP中附带）|每场景PNG/SVG；两张模式总览PNG/SVG|



只读取已存数值，不重新渲染。每个比较项的checkpoint路径、评价目录、seed、repeat、帧数和来源SHA256均保存在JSON。

Full地图N由PLY header核验；PSNR/SSIM/LPIPS由原始逐帧数值复算。
