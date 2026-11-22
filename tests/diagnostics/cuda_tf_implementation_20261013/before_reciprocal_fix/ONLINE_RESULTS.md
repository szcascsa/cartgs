# GradientConstraint关闭的完整在线结果

全部10次Mapping退出码0，50个Full/LoD评估退出码0；每组均评估全部输入关联帧，不抽帧、不训练测试帧。
同组仅backend不同；desk补充两边共同将TF start改为1000。tail_completed_steps=post_slam_mappings=0。
完整Mapping均值/p95是原验收executable的host cycle wall time；未使用诊断TF executable替代。
GPU显存为nvidia-smi每秒进程采样峰值，包含SLAM；不是Torch allocator peak。

## 原fast，TF start=3000

| 场景 | Backend | Mapping cycles | committed TF updates | CD TF steps | Gaussian total | Mapping mean/p95 ms | 进程wall s | 显存采样峰MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| long | legacy | 5431 | 2431 | 0 | 186219 | 21.2525/44.6500 | 127.702 | 3066 |
| long | cuda | 7420 | 4420 | 0 | 123070 | 15.7499/29.8394 | 128.967 | 2620 |
| room | legacy | 3790 | 790 | 0 | 148879 | 15.0248/33.7241 | 64.852 | 2632 |
| room | cuda | 4257 | 1257 | 55 | 141873 | 13.3467/28.8236 | 64.614 | 2688 |
| desk | legacy | 2188 | 0 | 0 | 56186 | 11.2351/19.0023 | 31.885 | 1250 |
| desk | cuda | 2220 | 0 | 0 | 65706 | 11.4309/17.6598 | 32.740 | 1342 |
| xyz | legacy | 8165 | 5165 | 20 | 26628 | 17.6104/25.2485 | 156.939 | 1070 |
| xyz | cuda | 13611 | 10611 | 20 | 30724 | 10.5759/12.9876 | 157.401 | 1148 |

| 场景 | Backend | translation APE RMSE m | rotation APE RMSE，unit-less |
|---|---|---:|---:|
| long | legacy | 0.009550 | 0.014292 |
| long | cuda | 0.032106 | 0.023336 |
| room | legacy | 0.070285 | 0.062111 |
| room | cuda | 0.064168 | 0.059782 |
| desk | legacy | 0.014653 | 0.038809 |
| desk | cuda | 0.016351 | 0.040666 |
| xyz | legacy | 0.002879 | 0.033390 |
| xyz | cuda | 0.002986 | 0.034779 |

APE使用SE(3) Umeyama alignment，未做scale alignment。

| 场景 | Backend | 口径 | 帧数 | PSNR dB | SSIM | LPIPS | active count | TF applied |
|---|---|---|---:|---:|---:|---:|---:|---|
| long | legacy | full | 2515 | 20.318686 | 0.729465 | 0.283790 | 186219 | false |
| long | legacy | lod_01 | 2515 | 18.128842 | 0.657809 | 0.518054 | 2323 | true |
| long | legacy | lod_05 | 2515 | 19.493417 | 0.697578 | 0.399975 | 9255 | true |
| long | legacy | lod_10 | 2515 | 19.947563 | 0.713617 | 0.346582 | 18511 | true |
| long | legacy | lod_15 | 2515 | 20.023979 | 0.716687 | 0.331819 | 27774 | true |
| long | cuda | full | 2515 | 17.632948 | 0.645101 | 0.407813 | 123070 | false |
| long | cuda | lod_01 | 2515 | 16.591930 | 0.620333 | 0.614992 | 1348 | true |
| long | cuda | lod_05 | 2515 | 17.165268 | 0.633617 | 0.494801 | 6196 | true |
| long | cuda | lod_10 | 2515 | 17.458037 | 0.640482 | 0.447707 | 12394 | true |
| long | cuda | lod_15 | 2515 | 17.486453 | 0.641566 | 0.434244 | 17452 | true |
| room | legacy | full | 1352 | 15.954337 | 0.598178 | 0.575147 | 148879 | false |
| room | legacy | lod_01 | 1352 | 14.230376 | 0.556592 | 0.706881 | 1753 | true |
| room | legacy | lod_05 | 1352 | 15.245139 | 0.580214 | 0.649468 | 7010 | true |
| room | legacy | lod_10 | 1352 | 15.662377 | 0.590389 | 0.618793 | 14894 | true |
| room | legacy | lod_15 | 1352 | 15.782167 | 0.593319 | 0.605341 | 22402 | true |
| room | cuda | full | 1352 | 16.106986 | 0.598520 | 0.571990 | 141873 | false |
| room | cuda | lod_01 | 1352 | 14.806798 | 0.566462 | 0.692404 | 2228 | true |
| room | cuda | lod_05 | 1352 | 15.638050 | 0.585363 | 0.646893 | 6846 | true |
| room | cuda | lod_10 | 1352 | 15.975560 | 0.593719 | 0.612006 | 14229 | true |
| room | cuda | lod_15 | 1352 | 16.070538 | 0.595742 | 0.599004 | 20888 | true |
| desk | legacy | full | 592 | 18.313167 | 0.664426 | 0.441846 | 56186 | false |
| desk | legacy | lod_01 | 592 | 16.306192 | 0.606740 | 0.585220 | 1514 | false |
| desk | legacy | lod_05 | 592 | 17.097743 | 0.625533 | 0.543419 | 2653 | false |
| desk | legacy | lod_10 | 592 | 17.600121 | 0.640228 | 0.506034 | 5083 | false |
| desk | legacy | lod_15 | 592 | 17.786812 | 0.645023 | 0.492714 | 7376 | false |
| desk | cuda | full | 592 | 18.393970 | 0.665613 | 0.435067 | 65706 | false |
| desk | cuda | lod_01 | 592 | 16.519777 | 0.607912 | 0.578382 | 1472 | false |
| desk | cuda | lod_05 | 592 | 17.265376 | 0.628841 | 0.528082 | 2845 | false |
| desk | cuda | lod_10 | 592 | 17.827512 | 0.645785 | 0.489728 | 5727 | false |
| desk | cuda | lod_15 | 592 | 17.985092 | 0.650660 | 0.473962 | 8538 | false |
| xyz | legacy | full | 3397 | 21.317513 | 0.716229 | 0.288390 | 26628 | false |
| xyz | legacy | lod_01 | 3397 | 17.364782 | 0.596245 | 0.678077 | 398 | true |
| xyz | legacy | lod_05 | 3397 | 18.996772 | 0.632377 | 0.524083 | 1304 | true |
| xyz | legacy | lod_10 | 3397 | 19.879330 | 0.659392 | 0.442679 | 2690 | true |
| xyz | legacy | lod_15 | 3397 | 20.090458 | 0.668233 | 0.417070 | 3576 | true |
| xyz | cuda | full | 3397 | 21.983111 | 0.731446 | 0.266643 | 30724 | false |
| xyz | cuda | lod_01 | 3397 | 17.494934 | 0.598159 | 0.683913 | 331 | true |
| xyz | cuda | lod_05 | 3397 | 19.247957 | 0.637075 | 0.527622 | 1386 | true |
| xyz | cuda | lod_10 | 3397 | 20.392369 | 0.671840 | 0.420552 | 3008 | true |
| xyz | cuda | lod_15 | 3397 | 20.636907 | 0.680094 | 0.396547 | 3823 | true |

### 生产Mapping分阶段采样

下表为既有MappingProfiler的current-stream CUDA event每19轮采样均值；阶段范围互有包含，不能逐列相加。host mean/p95见上表。

| 场景 | Backend | Full render ms | Full backward ms | LoD总范围 ms | LoD render ms | LoD backward范围 ms | 全optimizer ms | 总cycle CUDA ms | 样本数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| long | legacy | 0.9326 | 1.9222 | 13.4091 | 0.3766 | 8.1733 | 1.5620 | 20.5902 | 285 |
| long | cuda | 0.9566 | 1.9488 | 9.0093 | 0.3662 | 4.4572 | 1.1453 | 16.1938 | 390 |
| room | legacy | 0.6136 | 1.4596 | 8.6801 | 0.3083 | 5.1591 | 1.2364 | 14.9111 | 199 |
| room | cuda | 0.6429 | 1.4944 | 7.0082 | 0.3052 | 3.7379 | 1.1056 | 13.0900 | 224 |
| desk | legacy | 0.6644 | 1.5412 | 5.7373 | 0.3171 | 3.0135 | 1.0404 | 11.4979 | 115 |
| desk | cuda | 0.6681 | 1.5431 | 5.8962 | 0.3155 | 3.0200 | 1.0007 | 10.7515 | 116 |
| xyz | legacy | 0.7459 | 1.6341 | 12.5640 | 0.3376 | 8.6545 | 1.7119 | 17.5473 | 429 |
| xyz | cuda | 0.7956 | 1.7225 | 6.0782 | 0.3339 | 3.4853 | 1.0854 | 10.6207 | 716 |

### 逐1000轮的TF population观察

这些是原生产日志的周期观察，不是逐轮mean N。ratio固定为该轮抽到的0.01；其他ratio或视角不能由此推断。

| 场景 | Backend | mapping_iter:TF N |
|---|---|---|
| long | legacy | 1000:0, 2000:0, 3000:0, 4000:1426, 5000:1876 |
| long | cuda | 1000:0, 2000:0, 3000:0, 4000:1087, 5000:1515, 6000:1601, 7000:1284 |
| room | legacy | 1000:0, 2000:0, 3000:0 |
| room | cuda | 1000:0, 2000:0, 3000:0, 4000:1871 |
| desk | legacy | 1000:0, 2000:0 |
| desk | cuda | 1000:0, 2000:0 |
| xyz | legacy | 1000:0, 2000:0, 3000:0, 4000:265, 5000:413, 6000:374, 7000:394, 8000:348 |
| xyz | cuda | 1000:0, 2000:0, 3000:0, 4000:168, 5000:194, 6000:179, 7000:251, 8000:205, 9000:219, 10000:202, 11000:256, 12000:263, 13000:281 |

## 授权desk补充，TF start=1000

| 场景 | Backend | Mapping cycles | committed TF updates | CD TF steps | Gaussian total | Mapping mean/p95 ms | 进程wall s | 显存采样峰MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| desk | legacy | 1508 | 508 | 0 | 70014 | 16.1793/33.1772 | 31.733 | 1468 |
| desk | cuda | 1786 | 786 | 0 | 70313 | 13.8930/26.8303 | 32.543 | 1372 |

| 场景 | Backend | translation APE RMSE m | rotation APE RMSE，unit-less |
|---|---|---:|---:|
| desk | legacy | 0.015021 | 0.036909 |
| desk | cuda | 0.015563 | 0.040822 |

APE使用SE(3) Umeyama alignment，未做scale alignment。

| 场景 | Backend | 口径 | 帧数 | PSNR dB | SSIM | LPIPS | active count | TF applied |
|---|---|---|---:|---:|---:|---:|---:|---|
| desk | legacy | full | 592 | 16.938912 | 0.621312 | 0.489928 | 70014 | false |
| desk | legacy | lod_01 | 592 | 12.020828 | 0.494393 | 0.666284 | 1632 | true |
| desk | legacy | lod_05 | 592 | 13.755042 | 0.531267 | 0.605361 | 2890 | true |
| desk | legacy | lod_10 | 592 | 15.182681 | 0.565881 | 0.569629 | 4590 | true |
| desk | legacy | lod_15 | 592 | 15.740634 | 0.581540 | 0.547579 | 6296 | true |
| desk | cuda | full | 592 | 18.006290 | 0.655291 | 0.468875 | 70313 | false |
| desk | cuda | lod_01 | 592 | 16.137420 | 0.599940 | 0.593204 | 1733 | true |
| desk | cuda | lod_05 | 592 | 16.791949 | 0.617411 | 0.563714 | 3115 | true |
| desk | cuda | lod_10 | 592 | 17.284816 | 0.629638 | 0.535923 | 5518 | true |
| desk | cuda | lod_15 | 592 | 17.450364 | 0.635554 | 0.519046 | 10825 | true |

### 生产Mapping分阶段采样

下表为既有MappingProfiler的current-stream CUDA event每19轮采样均值；阶段范围互有包含，不能逐列相加。host mean/p95见上表。

| 场景 | Backend | Full render ms | Full backward ms | LoD总范围 ms | LoD render ms | LoD backward范围 ms | 全optimizer ms | 总cycle CUDA ms | 样本数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| desk | legacy | 0.6610 | 1.4842 | 9.9051 | 0.3287 | 6.2313 | 1.4178 | 16.4172 | 79 |
| desk | cuda | 0.7867 | 1.5189 | 7.4974 | 0.3454 | 4.1324 | 1.2926 | 13.2334 | 94 |

### 逐1000轮的TF population观察

这些是原生产日志的周期观察，不是逐轮mean N。ratio固定为该轮抽到的0.01；其他ratio或视角不能由此推断。

| 场景 | Backend | mapping_iter:TF N |
|---|---|---|
| desk | legacy | 1000:0 |
| desk | cuda | 1000:0 |
