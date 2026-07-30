#!/usr/bin/env python3
"""
Combined cost measurement script.

Runs one dataset sample through the full SPARC pipeline and both baselines
for all specified models, recording wall-clock time and token usage per agent.
Output is written to results/cost/ for use in the paper's cost tables.
"""

import sys
import os
import time
import json
import traceback
import signal
import contextlib

import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import *
from model import Model
from baselines import baseline_5_cot, baseline_7_sympy
from make_update_agent import update_agent
from state import NgSpiceState, run_ngspice_node
from creator import Creator
from agents.circuit_specification_agent import circuit_specification_agent_prompt
from agents.analysis_specification_agent import analysis_specification_agent_prompt
from agents.output_specification_agent import output_specification_agent_prompt
from agents.planner_agent import planner_agent
from agents.answer_generation_agent import answer_generation_agent

# ── Configuration ────────────────────────────────────────────────────────────

DATASET_CSV  = '../datasets/cktbench/cktbench.csv'
IMAGE_PATH   = '../datasets/cktbench/images'
OUTPUT_DIR   = 'results/cost'
SAMPLE_INDEX = 0          # single sample to test
TIMEOUT      = 8 * 60     # 8 minutes per pipeline run

# model-name → display-name (for the output table)
MODELS = {
    'openai':  'GPT-5.1',
    'claude':  'Claude Sonnet 4',
    'gpt4o':   'GPT-4o',
    'glm':     'GLM-4.5v',
    'qwen32b': 'Qwen3-VL-32B-Instruct',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds):
    def handler(signum, frame):
        raise TimeoutException(f"Timed out after {seconds} seconds")
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def reset_tokens(model):
    model.total_input_tokens = 0
    model.total_output_tokens = 0


def snap_tokens(model):
    return {'input': model.total_input_tokens, 'output': model.total_output_tokens}


# ── SPARC pipeline with per-agent instrumentation ────────────────────────────

def run_sparc(index, df, model):
    """
    Run the SPARC pipeline on one sample and return per-agent timing + tokens.

    Returns dict:
      {
        'planner':        {'time': s, 'tokens': {input, output}},
        'circuit_spec':   {'time': s, 'tokens': {input, output}},
        'analysis_spec':  {'time': s, 'tokens': {input, output}},
        'output_spec':    {'time': s, 'tokens': {input, output}},
        'answer_gen':     {'time': s, 'tokens': {input, output}},
        'total_time':     s,
        'error':          None or str,
      }
    """
    image = Image.open(f"{IMAGE_PATH}/{index}.png")
    row = df.iloc[index]

    netlist_val = row.get('netlist', '') if 'netlist' in df.columns else ''
    if pd.isna(netlist_val):
        netlist_val = ''

    metrics = {'error': None}
    wall_total_start = time.time()

    try:
        with time_limit(TIMEOUT):
            # ── Initialise state (mirrors Router.__init__) ────────────────
            if not netlist_val:
                creator = Creator(row['schema'])
                creator.make_template()
                state = NgSpiceState(model, creator)
            else:
                state = NgSpiceState(model)
                state.netlist_text = netlist_val

            state.index          = index
            state.question       = row['problem']
            state.schema         = row['schema']
            state.image          = image
            state.domain_knowledge = ''

            # ── Planner ──────────────────────────────────────────────────
            reset_tokens(model)
            t0 = time.time()
            sim_states = planner_agent(state)
            t_plan = time.time() - t0
            tok_plan = snap_tokens(model)
            metrics['planner'] = {'time': t_plan, 'tokens': tok_plan}

            t_circuit_total = 0.0;  tok_circuit = {'input': 0, 'output': 0}
            t_analysis_total = 0.0; tok_analysis = {'input': 0, 'output': 0}
            t_output_total = 0.0;   tok_output   = {'input': 0, 'output': 0}

            for sim_state in sim_states:
                # ── Circuit specification ─────────────────────────────────
                reset_tokens(model)
                t0 = time.time()
                sim_state = update_agent(sim_state, circuit_specification_agent_prompt,
                                         self_consistency_number=1)
                t_circuit_total += time.time() - t0
                t = snap_tokens(model)
                tok_circuit['input']  += t['input']
                tok_circuit['output'] += t['output']

                # ── Analysis specification ────────────────────────────────
                reset_tokens(model)
                t0 = time.time()
                sim_state = update_agent(sim_state, analysis_specification_agent_prompt)
                t_analysis_total += time.time() - t0
                t = snap_tokens(model)
                tok_analysis['input']  += t['input']
                tok_analysis['output'] += t['output']

                # ── Output specification ──────────────────────────────────
                reset_tokens(model)
                t0 = time.time()
                sim_state = update_agent(sim_state, output_specification_agent_prompt)
                t_output_total += time.time() - t0
                t = snap_tokens(model)
                tok_output['input']  += t['input']
                tok_output['output'] += t['output']

                # ── NGSpice (no API cost) ─────────────────────────────────
                sim_state = run_ngspice_node(sim_state)

            metrics['circuit_spec']  = {'time': t_circuit_total,  'tokens': tok_circuit}
            metrics['analysis_spec'] = {'time': t_analysis_total, 'tokens': tok_analysis}
            metrics['output_spec']   = {'time': t_output_total,   'tokens': tok_output}

            # ── Answer generation ─────────────────────────────────────────
            reset_tokens(model)
            t0 = time.time()
            answer_generation_agent(row['problem'], sim_states)
            t_ans = time.time() - t0
            tok_ans = snap_tokens(model)
            metrics['answer_gen'] = {'time': t_ans, 'tokens': tok_ans}

    except TimeoutException as e:
        metrics['error'] = f"TIMEOUT: {e}"
    except Exception as e:
        metrics['error'] = f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"

    metrics['total_time'] = time.time() - wall_total_start
    return metrics


# ── Baselines ─────────────────────────────────────────────────────────────────

def run_baseline(baseline_fn, model, *args):
    """Run a baseline function and return (wall_seconds, tokens, error)."""
    reset_tokens(model)
    t0 = time.time()
    error = None
    try:
        with time_limit(TIMEOUT):
            baseline_fn(model, *args)
    except TimeoutException as e:
        error = f"TIMEOUT: {e}"
    except Exception as e:
        error = f"ERROR: {type(e).__name__}: {e}"
    elapsed = time.time() - t0
    tokens = snap_tokens(model)
    return elapsed, tokens, error


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_agent_block(name, metrics_dict):
    if name not in metrics_dict:
        return f"  {name}: N/A\n"
    d = metrics_dict[name]
    t = d['time']
    tok = d['tokens']
    return (
        f"  {name}:\n"
        f"    time:         {t:.3f} s\n"
        f"    input tokens: {tok['input']}\n"
        f"    output tokens:{tok['output']}\n"
    )


def write_output(results, sample_index, df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, 'cost_results.txt')

    row = df.iloc[sample_index]
    with open(outpath, 'w') as f:
        f.write("=" * 72 + "\n")
        f.write("SPARC COST MEASUREMENT — single-sample run\n")
        f.write("=" * 72 + "\n")
        f.write(f"Sample index: {sample_index}\n")
        f.write(f"Question:     {row['problem'][:120]}\n")
        f.write("\n")

        # ── Per-model results ────────────────────────────────────────────
        for model_key, display_name in MODELS.items():
            if model_key not in results:
                continue
            res = results[model_key]
            f.write("-" * 72 + "\n")
            f.write(f"Model: {display_name}  ({model_key})\n")
            f.write("-" * 72 + "\n")

            sparc = res.get('sparc', {})
            b5    = res.get('baseline_cot', {})
            b7    = res.get('baseline_codegen', {})

            # SPARC breakdown
            f.write("SPARC pipeline:\n")
            if sparc.get('error'):
                f.write(f"  ERROR: {sparc['error']}\n")
            else:
                f.write(fmt_agent_block('circuit_spec',  sparc))
                f.write(fmt_agent_block('analysis_spec', sparc))
                f.write(fmt_agent_block('output_spec',   sparc))
                f.write(fmt_agent_block('answer_gen',    sparc))
                f.write(f"  SPARC total time: {sparc.get('total_time', 0):.3f} s\n")

            # Baselines
            f.write("Baseline — CoT (baseline_5):\n")
            if b5.get('error'):
                f.write(f"  ERROR: {b5['error']}\n")
            else:
                f.write(f"  time:         {b5.get('time', 0):.3f} s\n")
                tok = b5.get('tokens', {})
                f.write(f"  input tokens: {tok.get('input', 0)}\n")
                f.write(f"  output tokens:{tok.get('output', 0)}\n")

            f.write("Baseline — Codegen / SymPy (baseline_7):\n")
            if b7.get('error'):
                f.write(f"  ERROR: {b7['error']}\n")
            else:
                f.write(f"  time:         {b7.get('time', 0):.3f} s\n")
                tok = b7.get('tokens', {})
                f.write(f"  input tokens: {tok.get('input', 0)}\n")
                f.write(f"  output tokens:{tok.get('output', 0)}\n")

            f.write("\n")

        # ── Summary table (wall-clock times) ────────────────────────────
        f.write("=" * 72 + "\n")
        f.write("WALL-CLOCK TIME SUMMARY (seconds)\n")
        f.write("=" * 72 + "\n")
        header = f"{'Model':<28} {'SPARC':>8} {'Circ':>8} {'Anal':>8} {'Out':>8} {'CoT':>8} {'Cgen':>8}\n"
        f.write(header)
        f.write("-" * 80 + "\n")
        for model_key, display_name in MODELS.items():
            if model_key not in results:
                continue
            res = results[model_key]
            sparc = res.get('sparc', {})
            b5    = res.get('baseline_cot', {})
            b7    = res.get('baseline_codegen', {})
            sparc_t  = sparc.get('total_time', float('nan'))
            circ_t   = sparc.get('circuit_spec',  {}).get('time', float('nan'))
            anal_t   = sparc.get('analysis_spec', {}).get('time', float('nan'))
            out_t    = sparc.get('output_spec',   {}).get('time', float('nan'))
            cot_t    = b5.get('time', float('nan'))
            cgen_t   = b7.get('time', float('nan'))
            f.write(f"{display_name:<28} {sparc_t:>8.2f} {circ_t:>8.2f} {anal_t:>8.2f} {out_t:>8.2f} {cot_t:>8.2f} {cgen_t:>8.2f}\n")

        # ── Summary table (token usage — input) ─────────────────────────
        f.write("\n")
        f.write("=" * 72 + "\n")
        f.write("INPUT TOKEN SUMMARY\n")
        f.write("=" * 72 + "\n")
        f.write(header)
        f.write("-" * 80 + "\n")
        for model_key, display_name in MODELS.items():
            if model_key not in results:
                continue
            res = results[model_key]
            sparc = res.get('sparc', {})
            b5    = res.get('baseline_cot', {})
            b7    = res.get('baseline_codegen', {})
            def _itok(d, key):
                return d.get(key, {}).get('tokens', {}).get('input', 0) if isinstance(d.get(key), dict) else 0
            sparc_all = sum(_itok(sparc, k) for k in ['circuit_spec', 'analysis_spec', 'output_spec', 'answer_gen', 'planner'])
            circ_i  = _itok(sparc, 'circuit_spec')
            anal_i  = _itok(sparc, 'analysis_spec')
            out_i   = _itok(sparc, 'output_spec')
            cot_i   = b5.get('tokens', {}).get('input', 0)
            cgen_i  = b7.get('tokens', {}).get('input', 0)
            f.write(f"{display_name:<28} {sparc_all:>8} {circ_i:>8} {anal_i:>8} {out_i:>8} {cot_i:>8} {cgen_i:>8}\n")

        # ── Summary table (token usage — output) ────────────────────────
        f.write("\n")
        f.write("=" * 72 + "\n")
        f.write("OUTPUT TOKEN SUMMARY\n")
        f.write("=" * 72 + "\n")
        f.write(header)
        f.write("-" * 80 + "\n")
        for model_key, display_name in MODELS.items():
            if model_key not in results:
                continue
            res = results[model_key]
            sparc = res.get('sparc', {})
            b5    = res.get('baseline_cot', {})
            b7    = res.get('baseline_codegen', {})
            def _otok(d, key):
                return d.get(key, {}).get('tokens', {}).get('output', 0) if isinstance(d.get(key), dict) else 0
            sparc_all = sum(_otok(sparc, k) for k in ['circuit_spec', 'analysis_spec', 'output_spec', 'answer_gen', 'planner'])
            circ_o  = _otok(sparc, 'circuit_spec')
            anal_o  = _otok(sparc, 'analysis_spec')
            out_o   = _otok(sparc, 'output_spec')
            cot_o   = b5.get('tokens', {}).get('output', 0)
            cgen_o  = b7.get('tokens', {}).get('output', 0)
            f.write(f"{display_name:<28} {sparc_all:>8} {circ_o:>8} {anal_o:>8} {out_o:>8} {cot_o:>8} {cgen_o:>8}\n")

    print(f"\nResults written to: {outpath}")
    return outpath


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load API keys
    private_dir = '../private'
    try:
        with open(f'{private_dir}/azureapikey.txt') as fh:
            os.environ["AZURE_OPENAI_API_KEY"] = fh.read().strip()
        with open(f'{private_dir}/openrouterapikey.txt') as fh:
            os.environ["OPENROUTER_API_KEY"] = fh.read().strip()
    except FileNotFoundError as e:
        print(f"Warning: could not load API key: {e}")

    # Load dataset
    df = pd.read_csv(DATASET_CSV)
    print(f"Dataset loaded: {len(df)} rows. Using sample index {SAMPLE_INDEX}.")

    row = df.iloc[SAMPLE_INDEX]
    image_path = f"{IMAGE_PATH}/{SAMPLE_INDEX}.png"
    image = Image.open(image_path)
    schema  = row['schema']
    problem = row['problem']

    results = {}

    for model_key, display_name in MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {display_name}  ({model_key})")
        print('=' * 60)

        try:
            model = Model(model_key)
        except Exception as e:
            print(f"  Failed to initialise model: {e}")
            results[model_key] = {'error': str(e)}
            continue

        model_results = {}

        # ── SPARC ────────────────────────────────────────────────────────
        print("  Running SPARC pipeline …")
        sparc_metrics = run_sparc(SAMPLE_INDEX, df, model)
        model_results['sparc'] = sparc_metrics
        if sparc_metrics.get('error'):
            print(f"  SPARC error: {sparc_metrics['error']}")
        else:
            print(f"  SPARC total: {sparc_metrics['total_time']:.2f}s")

        # ── Baseline CoT ─────────────────────────────────────────────────
        print("  Running CoT baseline (baseline_5) …")
        t, tok, err = run_baseline(baseline_5_cot, model, problem, schema, image)
        model_results['baseline_cot'] = {'time': t, 'tokens': tok, 'error': err}
        if err:
            print(f"  CoT error: {err}")
        else:
            print(f"  CoT time: {t:.2f}s")

        # ── Baseline Codegen / SymPy ──────────────────────────────────────
        print("  Running Codegen baseline (baseline_7) …")
        t, tok, err = run_baseline(baseline_7_sympy, model, problem, schema, image)
        model_results['baseline_codegen'] = {'time': t, 'tokens': tok, 'error': err}
        if err:
            print(f"  Codegen error: {err}")
        else:
            print(f"  Codegen time: {t:.2f}s")

        results[model_key] = model_results

    # Write output
    outpath = write_output(results, SAMPLE_INDEX, df)

    # Also dump raw JSON for further analysis
    json_path = os.path.join(OUTPUT_DIR, 'cost_results_raw.json')
    with open(json_path, 'w') as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"Raw JSON written to: {json_path}")


if __name__ == "__main__":
    main()
