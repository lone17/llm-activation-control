import json

import numpy as np
import tinyBenchmarks as tb

# Choose benchmark (e.g. hellaswag)
benchmark = "hellaswag"  # possible benchmarks:
# ['mmlu','truthfulqa', 'gsm8k',
#  'winogrande', 'arc', 'hellaswag']

# Get score vector from output-file (the metric [here `acc_norm`] depends on the benchmark)
file_path = "benchmark/tinyBenchmarks/gemma-2-9b-it/baseline_vllm/google__gemma-2-9b-it/samples_tinyHellaswag_2025-03-12T20-33-05.308107.jsonl"
with open(file_path, "r", encoding="utf-8") as file:
    outputs = [json.loads(line) for line in file]

# Ensuring correct order of outputs
outputs = sorted(outputs, key=lambda x: x["doc_id"])

y = np.array([float(item["acc_norm"]) for item in outputs])

### Evaluation
tb.evaluate(y, benchmark)
