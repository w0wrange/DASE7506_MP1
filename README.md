# DASE7506 MP1: QK-Norm Language Model

Run the following PowerShell commands from this directory (`DASE7506_MP1/`). The course data, tokenizer, windowing protocol, and `evaluate.py` are unchanged. The final submitted model is a second-generation born-again student with width 348 and six attention heads; an earlier width-384 configuration exceeded the course CPU time limit. Detailed experiment records are in [RESULTS.md](reports/born-again-qknorm348-19200/RESULTS.md).

The baseline and final model use the same course evaluation protocol. Lower BPB is better. CPU time and memory come from a same-machine baseline bracket around the final model's evaluation (Intel i7-12700H, CPU FP32, four PyTorch threads).

| Metric | Course baseline | Final QK-Norm model |
|---|---:|---:|
| Layers / width / attention heads | 4 / 128 / 4 | 6 / 348 / 6 |
| Training steps per model | 1,200 | 19,200 |
| Validation BPB ↓ | 2.071083639 | **1.450150992** |
| Test BPB ↓ | 2.101260438 | **1.468030350** |
| Validation CPU scoring time | 6.096 s | 28.657 s |
| CPU time relative to baseline | 1.000× | **4.701×** |
| Validation peak resident memory | 1.567 GiB | 1.652 GiB |

Validation BPB decreases by **0.620933** (about 29.98%), and test BPB by **0.633230** (about 30.14%). For timing, the baseline denominator is the faster of the two bracketing runs; for baseline memory, the larger of their two peaks is reported. The final model was scored between those baseline runs. Complete inference assets occupy **34,743,281 bytes**, within the course limits of CPU time ≤5× baseline, RAM ≤4 GiB, and assets ≤64 MiB. The first formal test score was computed only after validation-based model selection was frozen. A later independent rerun of the same model checked reproducibility and was not used for selection. Sources: [baseline report](reports/baseline-s17/RESULTS.md), [final summary](reports/born-again-qknorm348-19200/final-results.json), and [same-machine bracket record](reports/born-again-qknorm348-19200/born-again-fast/result.json).

## 1. Install

Use Python 3.12. `requirements.txt` pins PyTorch 2.7.1, NumPy 2.5.3, and tokenizers 0.21.4. To evaluate the supplied model, a CPU environment is sufficient:

```powershell
python --version
python -m venv .venv-cpu
.\.venv-cpu\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-cpu\Scripts\python.exe -m pip install -r requirements.txt
```

Full retraining also requires an NVIDIA GPU. The original experiments used the CUDA 12.6 build of PyTorch 2.7.1. Create a separate `.venv` and confirm that the GPU is available. If your machine needs a different CUDA build, adjust the wheel index according to the [official PyTorch version guide](https://pytorch.org/get-started/previous-versions/):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The `data/` directory contains the course-provided train, validation, and test text, together with the tokenizer.

Data attribution: the course-provided text comes from WikiText-2, introduced by [Merity et al. in *Pointer Sentinel Mixture Models*](https://arxiv.org/abs/1609.07843), with original text from Wikipedia. See the [WikiText dataset page](https://huggingface.co/datasets/Salesforce/wikitext) for data provenance and licensing information.

## 2. Train and evaluate

The starting teacher weights are retained at `runs/independent-teacher-9600-s71/checkpoint-average.pt` and `runs/extend-8400-fastcache-s29/checkpoint.pt`. Two first-generation students learn from these teachers; the second-generation student learns from the first-generation pair. A matched direct-distillation control learns from the original teachers. All four students start from random weights and each trains for 19,200 steps with batch size 32 and context length 256.

Training uses FP32, AdamW, dropout 0.05, and weight decay 0.05. The learning rate peaks at 0.001, warms up for 100 steps, and then decays with a cosine schedule to 0.0001. Hard-label and teacher soft-label losses have equal weight. The two teachers' probabilities are computed with separate softmax operations and then averaged. The final weights average the checkpoints at steps 17,760, 18,240, 18,720, and 19,200. The following commands write to a new `runs/reproduction-qknorm348` directory. The two finalization scripts use only validation data for calibration and selection and check CPU resources:

```powershell
$runRoot = 'runs\reproduction-qknorm348'
$report = 'reports\reproduction-qknorm348'
.\.venv\Scripts\python.exe tools\run_born_again_19200.py --config configs\rope_qknorm_w348_h6_d6.json --run-root $runRoot
.\.venv\Scripts\python.exe tools\finalize_born_again_19200.py --run-root $runRoot
.\.venv\Scripts\python.exe tools\finalize_qknorm_fast.py --run-root $runRoot --report $report
```

Use an unused `runRoot`. The directory name of `report` must match that of `runRoot`. After finalization, if `frozen-selection.json` exists, use its recorded checkpoint path to rescore validation:

```powershell
$selected = (Get-Content "$report\frozen-selection.json" -Raw | ConvertFrom-Json).checkpoint
.\.venv-cpu\Scripts\python.exe evaluate.py --checkpoint $selected --device cpu --precision fp32 --threads 4 --split validation --output "$runRoot\validation.json"
```

Validation determines the final method. Evaluate test only after freezing the selection; do not tune using test results. The scorer also saves per-window losses in a `.window-nll.npy` file beside the JSON output.

## 3. Reproduction

Evaluate the frozen [final checkpoint](reports/born-again-qknorm348-19200/born-again-fast/checkpoint.pt) without retraining:

```powershell
$checkpoint = 'reports\born-again-qknorm348-19200\born-again-fast\checkpoint.pt'
.\.venv-cpu\Scripts\python.exe evaluate.py --checkpoint $checkpoint --device cpu --precision fp32 --threads 4 --split validation --output runs\reproduced-validation.json
.\.venv-cpu\Scripts\python.exe -m unittest discover -s tests -q
```

Checkpoint SHA-256: `981d85a72df78ffa0a53845f1e00f2ddf0334c40682afaa7ad0d476274def681`. Recomputed CPU FP32 validation BPB should be close to **1.450150992**. The first formal test score after freezing the selection is in [selected-test.json](reports/born-again-qknorm348-19200/selected-test.json); the later independent rerun is in [reproduced-test.json](reports/born-again-qknorm348-19200/reproduced-test.json). Both runs have identical results and per-window losses across all 1,674 windows. The [frozen selection record](reports/born-again-qknorm348-19200/frozen-selection.json) and [final summary](reports/born-again-qknorm348-19200/final-results.json) record the checkpoint fingerprint, evaluation results, and resource checks. The `test_evaluated_once` field in `final-results.json` describes the state at the time of the first formal score. CPU timing varies with hardware and system load and should be compared against a baseline run on the same machine.

The final student uses FFN width 688, QK-Norm, RoPE, SwiGLU, and a causal cache. The inference implementation optimizes only the RoPE computation; it does not change the weights or validation BPB. The matched direct-distillation control has a slightly better validation BPB of 1.448343301, but its CPU time was 5.366× the baseline in the recorded bracket, above the course limit. The second-generation student was therefore selected. This experiment does not establish that multigeneration distillation improves prediction quality. Training used only the course training text; calibration and model selection used only validation data. OpenAI Codex provided substantive assistance with the model code and organization of the results.
