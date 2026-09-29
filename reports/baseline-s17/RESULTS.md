# Project 1 baseline 实验记录

运行日期：2026-09-18（UTC+8）；run ID：`baseline-s17`。

完整 baseline 已完成：validation **2.071083638533899 BPB**，test **2.1012604380869244 BPB**。5 项 contract tests 全部通过，依赖检查通过。

## 实验设置

- 原始 `model.py`，协议 `7506-mp1-wt2-v2`，从随机初始化训练，无父 checkpoint。
- 4 层 GPT，width 128、4 个 attention heads、context 256、词表 2048；参数量 1,088,256。
- CPU、FP32、4 个 PyTorch 线程；seed 17；1,200 步，batch size 32。
- 处理训练目标数：1,200 × 32 × 256 = **9,830,400**。
- 固定使用第 1,200 步 checkpoint；本次未做超参数搜索或根据 test 选择模型。
- 硬件：Intel Core i7-12700H（20 个逻辑处理器）；Windows 11，build 26200。
- Python 3.12.2；PyTorch 2.7.1+cpu；NumPy 2.5.3；tokenizers 0.21.4。
- 训练执行时间：00:27:33–00:35:17（UTC+8）。

## 测量结果

| 阶段 | BPB | 纯训练/评分耗时（秒） | 含导入、加载的总耗时（秒） | 峰值驻留 RAM（GiB） |
|---|---:|---:|---:|---:|
| 训练 | — | 443.043579 | 464.245594 | 1.557083 |
| 独立 validation 评分 | 2.071083639 | 6.458481 | 17.770384 | 1.569340 |
| 独立 test 评分 | 2.101260438 | 8.011105 | 17.677552 | 1.568562 |

训练总耗时还包含训练器自带的最终 validation（8.905178 秒）及 checkpoint 保存。该内置 validation 与独立评分的 BPB 完全一致。

多线程累计 CPU 用时分别为 1,489.562500 / 33.578125 / 36.390625 秒（训练 / validation / test）；这是各线程 CPU 用时之和，不是表中的实际等待时间，也不是作业 scoring wall time。

RAM 由 [测量脚本](../../tools/measure_run.py) 调用 Windows `GetProcessMemoryInfo`，读取操作系统维护的进程生命周期最高工作集（PeakWorkingSetSize）。每次训练或独立评分使用一个全新 Python 进程，包含解释器、依赖导入、数据加载及执行开销；本 baseline 未启动子 worker。记录器不改动模型、数据或评分逻辑。

峰值提交内存另记为 2,421.531250 / 2,446.203125 / 2,445.457031 MiB。提交内存与实际驻留 RAM 是不同指标。原始 JSON 中的 `peak_allocated_gb=0` 和 `peak_reserved_gb=0` 是 CUDA 专用统计在 CPU 下的返回值，不能当作 CPU RAM。

每个 split 的独立评分时间仅测量一次。本机后续资源比较可暂用 test 8.011105 秒作为基准，其 5 倍约为 **40.055527 秒**；正式比较应在相同机器、线程数和相近负载下重新对照 baseline。评测峰值 RAM 低于 4 GiB。

## Checkpoint 与正确性核对

- [checkpoint.pt](../../runs/baseline-s17/checkpoint.pt)：4,370,805 字节，约 **4.168 MiB**。PyTorch ZIP 条目未压缩，内容合计 4,359,647 字节；没有额外学习得到的推理资产。该大小单独指 checkpoint，不包含共享评分器、tokenizer 和源代码。
- SHA256：`b7fdb67c8ef14b5166c0de1f17cf306f34800b41dfe97da856c9812cdb1f9701`。
- Validation：376,599 个目标、1,148,007 UTF-8 字节、1,472 个窗口。
- Test：428,405 个目标、1,292,013 UTF-8 字节、1,674 个窗口。
- 两份评分 JSON 使用相同 checkpoint；保存的逐窗口损失可重算得到对应 BPB。
- 模型、训练器、评分器、公共逻辑、配置、测试、依赖列表和数据文件哈希均匹配发布清单。
- 开始前已存在的文档差异：`README.md` 的哈希与发布清单不同，实际为 `38b196fc2d3f0f50fd8633ea638d4f199d66c2d301aa4f632300be90557f7ea2`；本次未改动该文档。
- 当前目录没有 Git 仓库元数据，本记录用发布清单哈希及 checkpoint 哈希标识版本。
- Codex 协助执行、编写资源测量脚本和整理本记录；原始 baseline 算法、训练脚本、评分器与数据均未修改。

## 原始结果与日志

- [机器可读汇总](summary.json)
- [训练指标及每 100 步 loss](../../runs/baseline-s17/metrics.json)
- [Validation 原始评分](../../runs/baseline-s17/validation_cpu_fp32.json) · [逐窗口损失](../../runs/baseline-s17/validation_cpu_fp32.window-nll.npy)
- [Test 原始评分](../../runs/baseline-s17/test_cpu_fp32.json) · [逐窗口损失](../../runs/baseline-s17/test_cpu_fp32.window-nll.npy)
- [测试日志](tests.log) · [训练日志](train.log) · [Validation 日志](validation.log) · [Test 日志](test.log)
- [测试资源](tests.resources.json) · [训练资源](train.resources.json) · [Validation 资源](validation.resources.json) · [Test 资源](test.resources.json)

## 复现命令

以下 PowerShell 命令复现相同设置，并使用新的 `baseline-s17-repeat` 路径以保留本次结果。再次复现时仍需选择未使用的路径。

```powershell
cd E:\HKU_course\DASE7506\project1\MP1_student_starter\MP1_student_starter\DASE7506_MP1

.\.venv\Scripts\python.exe -u tools\measure_run.py --record-prefix reports\baseline-s17-repeat\tests -- -m unittest discover -s tests -v

.\.venv\Scripts\python.exe -u tools\measure_run.py --record-prefix reports\baseline-s17-repeat\train -- train.py --implementation model --device cpu --precision fp32 --threads 4 --seed 17 --steps 1200 --batch-size 32 --run-dir runs\baseline-s17-repeat

.\.venv\Scripts\python.exe -u tools\measure_run.py --record-prefix reports\baseline-s17-repeat\validation -- evaluate.py --checkpoint runs\baseline-s17-repeat\checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation --output runs\baseline-s17-repeat\validation_cpu_fp32.json

.\.venv\Scripts\python.exe -u tools\measure_run.py --record-prefix reports\baseline-s17-repeat\test -- evaluate.py --checkpoint runs\baseline-s17-repeat\checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output runs\baseline-s17-repeat\test_cpu_fp32.json
```

