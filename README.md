# Benchmark evaluation script

Code to numerically evaluate benchmark dataset at https://huggingface.co/datasets/AnonBenchmark5727/benchmark_data

## To install
Install the package locally by entering the directory with `setup.py` and running
```
pip install -e .
```

## Configuration
Select models for evaluation by editing `config.json`.

Make sure that your directory contains a `.env` file that contains the API keys for the various model providers saved as:
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`

Within `scripts` the file `eval_config.json` configures which LLMs will be evalauted by toggling a given model `true`.

The file `src/benchmark_evaluator/config/model_config.json` defines the model ID that will be queried for a particular choice of model. Update this file as necessary if a particular snapshot is desired. Note that after updating this file you may need to reinstall the package.

## To run evaluations
To run the evaluation, in the main directory run
```
python scripts/run_eval.py
```
The complete results are saved in `results/full_results.json` while `results/results.json` has an abbreviated results that removes the prompt and does not save intermediate Sympy expressions within the `EvaluationResult` framework.

Once the evaluation is complete, you may run
```
python scripts/generate_summary.py
```
to generate summary statistics in `summary.json` and to generate figures summarizing the results.

You may also run
```
python scripts/generate_latex_summary.py
```
to generate two `.tex` files, `llm_summary.tex` which contains a summary of the LLM solutions and expected solutions as well as `llm_full_results.tex` which gives the full model response to the question.
