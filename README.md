# SPARC code

This repository contains the code and datasets for running the circuit diagram understanding agent and baseline experiments on two benchmarks: **CktBench** and **NetQ**.

---

## Prerequisites

### Python dependencies

Install required packages from the project root:

```bash
pip install openai anthropic pandas pillow pyyaml sentence-transformers numpy requests torch transformers
```

NGSpice must be installed for simulation-based methods:

```bash
sudo apt-get install ngspice   # Ubuntu/Debian
brew install ngspice           # macOS
```

### API keys

Set the following environment variables before running:

```bash
export AZURE_OPENAI_API_KEY="your-azure-openai-key"      # for --model openai (GPT-4o via Azure)
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"  # your Azure OpenAI resource endpoint
export AZURE_OPENAI_ORG="your-azure-org-id"              # optional, if your Azure resource requires an org id
export OPENROUTER_API_KEY="your-openrouter-key"           # for all other models (claude, gemini, qwen, etc.)
```

### Configuration

A `configs/config.yaml` file is expected at the project root (one level above `src/`). The config specifies dataset paths and any additional settings loaded by `utils.load_config()`.

---

## Datasets

| Dataset | File | Size | Description |
|---|---|---|---|
| CktBench | `datasets/cktbench/cktbench.csv` | 1205 problems | Circuit diagram MCQ and open-ended questions with netlisttics |
| NetQ | `datasets/netq/netq.csv` | 495 problems | Analog/mixed-signal circuit simulation questions with netlists |

---

## Running Experiments

All scripts are in `src/`. **Run them from the `src/` directory.**

```bash
cd src
```

### Main agent

**CktBench:**
```bash
python run.py --model openai
python run.py --model claude
```

**NetQ:**
```bash
python run_amsnet.py --model openai
python run_amsnet.py --model claude
```

**Supported `--model` values:** `openai`, `claude`, `gemini`, `qwen`, `qwen32b`, `llava`, `internvl`, `nemotron`, `glm`, `llama`

To run on specific indices only:
```bash
python run.py --model openai --indices "0,5,10,15"
```

---

### Baselines

| Baseline | Description |
|---|---|
| B4 | Chain-of-thought, netlist only |
| B5 | Chain-of-thought, netlist + image |
| B6 | Chain-of-thought, image only |
| B7 | SymPy code generation — model writes and executes Python/SymPy code to solve the circuit problem |

**CktBench:**
```bash
python run_baselines.py --model openai --base-dir files/run --result-dir results/run
```

**NetQ:**
```bash
python run_baselines_amsnet.py --model openai --base-dir files/run --result-dir results/run
```

---

### Direct simulation (NGSpice)

Runs NGSpice simulation directly from the netlist without agent reasoning.

**CktBench:**
```bash
python run_direct_simulation.py --model openai
```

**NetQ:**
```bash
python run_direct_simulation_amsnet.py --model openai
```

---

### Ablation studies

Run all 10 ablation variants (component removal: no circuit spec, no analysis spec, no output spec, no web search, no image) on both CktBench and NetQ:

```bash
python ablations/run_all_ablations.py
python ablations/run_all_ablations.py --concurrency 2   # lower if hitting rate limits
```

Individual ablations can also be run directly, e.g.:

```bash
python ablations/run_without_web.py
python ablations/run_amsnet_without_C.py
```

---

## Output

Results are written to a model-specific results directory (e.g., `results/run_openai/`). Each index produces a JSON file with the predicted answer, ground truth, and correctness flag. A summary file is written at the end of each run.

---

## Netlist Extraction (Segment–Trace–Reason)

`netlist_extraction/` turns a circuit schematic image into a structured netlist (components + connections). It implements the **Segment–Trace–Reason (STR)** pipeline described in the paper.

### Additional dependencies

```bash
pip install opencv-contrib-python easyocr
```

The `sam2` package (Meta's Segment Anything 2) is also required:

```bash
pip install sam2
```

Download the SAM2.1 small checkpoint and place it at `netlist_extraction/checkpoints/sam2.1_hiera_small.pt`:

```bash
mkdir -p netlist_extraction/checkpoints
curl -L -o netlist_extraction/checkpoints/sam2.1_hiera_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

### Running it

```bash
cd netlist_extraction

# Single image
python extract.py --image ../datasets/cktbench/images/0.png --model claude --out result.json

# Batch over a directory of images
python extract.py --image_dir ../datasets/cktbench/images --model openai --out results.json

# CPU-only (no GPU for SAM2)
python extract.py --image ../datasets/cktbench/images/0.png --model claude --device cpu
```

`--model` accepts the same backends as the main agent (`openai`, `claude`, `gemini`, `qwen`, `llava`, `internvl`, `nemotron`, `glm`); the same API keys from "Prerequisites" above apply. `--variant` overrides the backend's default model name/variant (e.g. `--model openai --variant claude-opus-4-7` if routed through a gateway).

Output is the parsed netlist JSON: `{"components": [{"id": ..., "type": ...}, ...], "connections": [[id_a, id_b], ...], "marker_mapping": {...}}`.

Files:
| File | Purpose |
|---|---|
| `extract.py` | Entry point — runs the full pipeline end to end (CLI + `extract_netlist()`). |
| `text_utils.py` | EasyOCR text detection/erasure (location only, never recognized content). |
| `sam_marking.py` | SAM2 zero-shot component segmentation → numbered Set-of-Mark image. |
| `wire_graph.py` | Deterministic wire tracing, junction repair, series/parallel/ground detection. |
| `prompts.py` | The joint-reasoning prompt template and output JSON schema. |
| `json_utils.py` | Robust JSON extraction from raw VLM output. |

---

## Repository Structure

```
submission/
├── datasets/
│   ├── cktbench/cktbench.csv     # CktBench benchmark
│   └── netq/netq.csv             # NetQ benchmark
├── netlist_extraction/           # Segment-Trace-Reason netlist extraction pipeline
│   ├── extract.py                # Entry point — image -> netlist, end to end
│   ├── text_utils.py             # EasyOCR text detect/erase
│   ├── sam_marking.py            # SAM2 component segmentation
│   ├── wire_graph.py             # Deterministic wire tracing / connectivity
│   ├── prompts.py                # Joint-reasoning prompt template
│   ├── json_utils.py             # VLM output JSON parsing
│   └── checkpoints/              # SAM2 checkpoint (downloaded, not versioned)
└── src/
    ├── run.py                    # Main agent — CktBench
    ├── run_amsnet.py             # Main agent — NetQ
    ├── run_baselines.py          # Baselines — CktBench
    ├── run_baselines_amsnet.py   # Baselines — NetQ
    ├── run_direct_simulation.py  # NGSpice direct sim — CktBench
    ├── run_direct_simulation_amsnet.py  # NGSpice direct sim — NetQ
    ├── ablations/                # Ablation study scripts
    ├── agents/                   # Agent modules (planner, circuit spec, analysis spec, etc.)
    ├── router.py                 # Agent orchestration
    ├── model.py                  # Model API wrappers
    ├── evaluation.py             # Correctness scoring
    ├── baselines.py              # Baseline implementations
    └── utils.py                  # Shared utilities
```
