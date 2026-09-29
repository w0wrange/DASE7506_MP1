# QK-Norm 六头多代蒸馏：最终结果

课程基线 CPU FP32 验证 BPB 为 2.071083639、test BPB 为
2.101260438；本实验此前最好的验证 BPB 为 1.468614158。
模型、tokenizer、评估器和数据均采用课程固定协议。训练与选择只使用
训练集和验证集；最终方法冻结后才进行首次正式 test 评分，之后对同一模型独立重跑一次用于复现核验。两次 test 结果均未用于选模。

## 结论

最终提交模型为 [第二代 born-again 推理权重](born-again-fast/checkpoint.pt)。
模型为 6 层、宽度 348、6 头、FFN 688，QK-Norm、RoPE、SwiGLU，
dropout 0.05，weight decay 0.05。每位学生从随机权重训练 19,200 步；
384 宽、6 头的要求配置此前实测超过课程 5 倍 CPU 耗时上限，
因此实际训练采用 348 宽；未采用的配置已移至历史归档。

| 指标 | 最终结果 | 限制或对照 |
|---|---:|---:|
| CPU FP32 验证 BPB | **1.450150992** | 原最佳 1.468614158；改善 0.018463166 |
| CPU 验证耗时 | 28.657 秒 | 同次最快基线 6.096 秒；**4.701 倍 ≤ 5 倍** |
| 验证峰值驻留内存 | 1.652 GiB | ≤ 4 GiB |
| 完整推理资产 | 34,743,281 字节 | ≤ 67,108,864 字节 |
| 冻结后首次正式 test 评估 BPB | **1.468030350** | CPU FP32；未用于选模 |
| test 峰值驻留内存 | 1.652 GiB | ≤ 4 GiB |

推理实现将相邻 RoPE 维度组成复数进行乘法，不改变模型权重、配置
或验证 BPB。完整验证集上，优化前后的 BPB 均为 1.4501509920851057。
优化实现源码为 `student_qknorm_rotary_inference.py`，其所依赖的训练架构
源码为 `student_qknorm_rotary.py`；两者均计入完整推理资产。

直接对照和第二代学生都使用 seed 37、宽度 348、6 头、相同批次与
19,200 步预算，各处理 157,286,400 个目标，因而隔离的是额外一代
教师蒸馏。第二代校准前验证缓存 BPB 比直接对照高 0.004513755；
校准后高 0.001807691。这次实验没有证明多代蒸馏带来质量收益。
最终选择第二代，是因为只有它通过同次最快基线的保守 CPU 限制。
两位第一代学生额外处理了 314,572,800 个目标，因而这种方案的
训练成本高于直接蒸馏。QK-Norm 与宽度变化没有单独配对消融，
不能将相对旧模型的改善单独归因于 QK-Norm。

## 训练与选择

四位学生各训练 19,200 步、batch 32、context 256，合计
629,145,600 个训练目标。第一代两位学生使用原有两位教师；第二代
使用第一代两位学生；直接蒸馏对照使用原教师且与第二代同为 seed 37。
教师仅用神经网络概率，双教师概率算术平均，硬/软损失各占一半。
平均权重取 17,760、18,240、18,720 和 19,200 步。下表为这些平均
权重在训练配方默认缓存下的验证集结果：

| 学生 | 神经网络 BPB | 缓存 BPB |
|---|---:|---:|
| 第一代 seed 29 | 1.477216918 | 1.449055577 |
| 第一代 seed 31 | 1.482422620 | 1.452581844 |
| 第二代 seed 37 | 1.488795358 | 1.457722246 |
| 直接对照 seed 37 | 1.480346640 | 1.453208490 |

仅在验证集上从晚期检查点和平均权重中选源，并按预设网格校准。
最终校准为神经温度 1.1、缓存 theta 13、缓存权重 0.05。
推理优化后的直接对照验证 BPB 为 1.448343301，但与同次最快基线
相比耗时 5.366 倍，不满足保守的资源判定；第二代通过全部上限，
因此在验证阶段冻结为最终方法。`frozen-selection.json` 保存冻结时的
方法、权重指纹和资源测量；`optimized-summary.json` 保留两个候选结果。
对照模型权重和未选中的 CPU 运行产物已移至同级历史归档；提交目录
保留数值、资源记录和最终模型的完整验证产物。

冻结后先对选中权重运行一次正式 test 评分。`selected-test.json` 保存该次
CPU FP32 结果，`selected-test.resources.json` 保存该进程的内存与时间。
随后独立重跑同一权重和实现，结果保存在 `reproduced-test.json`；两次 BPB
及全部 1,674 个窗口的损失值逐项一致。`final-results.json` 汇总首次正式
评分时的决定，其中 `test_evaluated_once` 指当时的状态。测试集结果没有
反向影响模型、校准设置或阈值。

时间记录说明：`frozen-selection.json` 与 `optimized-summary.json` 在本机
2026-09-29 01:22:09（UTC+8）创建，首次 test 进程于 01:22:32 启动。
两份选择文件的当前修改时间为 01:44:06；收尾脚本在确认已有冻结选择
内容完全一致后仍会重写文件，因此不能单凭当前修改时间判断选模先后。
首次 test 的启动时间、权重及实现指纹保存在 `selected-test.resources.json`
和 `selected-test.json` 中。进程日志里的工作目录 `code` 是项目目录改名为
`DASE7506_MP1` 前的历史路径。

## 复现命令

```powershell
.\.venv\Scripts\python.exe tools\run_born_again_19200.py --config configs\rope_qknorm_w348_h6_d6.json --run-root runs\reproduction-qknorm348
.\.venv\Scripts\python.exe tools\finalize_born_again_19200.py --run-root runs\reproduction-qknorm348
.\.venv\Scripts\python.exe tools\finalize_qknorm_fast.py --run-root runs\reproduction-qknorm348 --report reports\reproduction-qknorm348
```

以上选模脚本只读取验证集，不会评估 test。原实验训练与非最终对照
的详细产物保存在同级历史归档；此目录保留最终权重、验证和 test 记录。
test 命令及进程记录见 `selected-test.resources.json`。
