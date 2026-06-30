# Fine-Tuning Small Models Locally and Evaluation

This repository teaches how to prepare an instruction dataset, fine-tune a
small language model locally with
[LoRA](https://arxiv.org/abs/2106.09685), compare the adapted model against the
base model, and evaluate open-ended answers with local metrics plus an optional
[LLM-as-judge](https://arxiv.org/abs/2306.05685) workflow. The notebooks use
[Hugging Face Transformers](https://huggingface.co/docs/transformers/index),
[Hugging Face PEFT](https://huggingface.co/docs/peft/index), and a small
support-policy dataset so the complete workflow stays inspectable on a laptop.
The runnable path fine-tunes
[SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct),
a compact public model that is large enough to show an observable adapter
effect while remaining practical for a local teaching exercise.

## Learning Path

```mermaid
flowchart LR
    A["01 - Instruction data"] --> B["02 - LoRA and QLoRA"]
    B --> C["03 - Local fine-tuning"]
    C --> D["04 - Base vs adapted eval"]
    D --> E["05 - Judge and faithfulness"]
```

- [01 - Instruction dataset readiness](01-instruction-dataset-readiness.ipynb): Convert domain examples into completion and chat-style records, validate fields, and split a held-out evaluation set.
- [02 - LoRA and QLoRA mechanics](02-lora-qlora-mechanics.ipynb): Inspect why adapter training changes a small number of parameters and how quantization changes memory trade-offs.
- [03 - Local LoRA fine-tuning run](03-local-lora-fine-tuning-run.ipynb): Fine-tune a small causal language model locally and save adapter weights to an ignored output folder.
- [04 - Base vs fine-tuned evaluation](04-base-vs-finetuned-evaluation.ipynb): Compare base and adapter responses with held-out prompts, ROUGE, BLEU, perplexity, and error analysis.
- [05 - LLM-as-judge and faithfulness checks](05-llm-as-judge-and-faithfulness.ipynb): Use optional Groq/Mistral judge calls, structured scoring, and source-faithfulness checks.

## Training Scope

Notebook 03 performs real local LoRA training and writes a real adapter
checkpoint. The intentionally small dataset demonstrates the workflow rather
than producing a deployable support model. Expect some metrics to improve while
individual policy answers remain incorrect; notebooks 04 and 05 use those
failures to teach evaluation, faithfulness, and the next data iteration.

QLoRA is explained but not executed because its main benefit appears when a
larger model must fit into limited accelerator memory. The runnable CPU path
uses ordinary LoRA to keep hardware requirements predictable.

## Exercise Format

The notebooks use small checkpoints throughout the learning path instead of one
large exercise at the end. Code exercises appear as short TODO cells followed
immediately by a runnable solution cell. Concept checks use brief text, image,
diagram, or output questions followed immediately by the answer or explanation.

## Repository Map

```text
mle-fine-tuning-small-models/
├── assets/
├── data/
│   ├── evaluation_cases.json
│   ├── reference_material.json
│   └── support_instruction_examples.jsonl
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── 01-05-*.ipynb
```

## Environment

Please make sure you **use this repository as a template** and set up a new virtual environment. You can use the following commands:

### macOS

```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

### Windows PowerShell

```powershell
pyenv local 3.11.3
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### Windows Git Bash

```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

The notebooks download a small public model from the
[Hugging Face Hub](https://huggingface.co/docs/hub/index) the first time the
model notebooks run. The model download is approximately 270 MB, and the
reference training loop normally takes around one to three minutes on a recent
laptop CPU. Adapter checkpoints and generated files are written to ignored
`outputs/` or `artifacts/` folders.

## Optional Judge Provider Setup

The local fine-tuning notebooks do not require hosted LLM calls. The final
evaluation notebook can use a stronger model as a judge through the
[OpenAI Python client](https://github.com/openai/openai-python), with
[Groq OpenAI compatibility](https://console.groq.com/docs/openai) as the
primary provider and
[Mistral chat completions](https://docs.mistral.ai/studio-api/conversations/chat-completion)
as the fallback provider.

When no API key is configured, the judge notebook runs in deterministic dry
mode and returns readable mock scores. To run live judging, copy `.env.example`
to `.env` and add at least one key:

- `LLM_PROVIDER=groq` uses Groq first.
- `GROQ_API_KEY` enables Groq calls through `https://api.groq.com/openai/v1`.
- `MISTRAL_API_KEY` enables Mistral calls through `https://api.mistral.ai/v1`
  and acts as a fallback when both keys are configured.
- `GROQ_MODEL` and `MISTRAL_MODEL` can override the default model names.

Provider calls may incur costs or hit rate limits. Do not commit `.env` or API
keys.

## Learning Objectives

By the end of this repository, you should be able to:

- Prepare and format a domain-specific instruction dataset for fine-tuning.
- Fine-tune a small language model locally with LoRA and explain where QLoRA changes the memory profile.
- Compare a fine-tuned adapter against the base model on a held-out evaluation set.
- Implement an LLM-as-judge evaluation pipeline for relevance, faithfulness, and coherence.
- Identify hallucination and source-faithfulness failures by comparing generated claims against reference material.

## Further Reading

- [LoRA paper](https://arxiv.org/abs/2106.09685): the original low-rank adapter method.
- [QLoRA paper](https://arxiv.org/abs/2305.14314): quantized fine-tuning with low-rank adapters.
- [Hugging Face PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora): practical adapter configuration.
- [Hugging Face Transformers Trainer](https://huggingface.co/docs/transformers/trainer): training and evaluation loop used in many fine-tuning workflows.
- [Hugging Face TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer): supervised fine-tuning trainer used for larger instruction-tuning projects.
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/index): dataset representation and transformations.
- [ROUGE](https://aclanthology.org/W04-1013/): overlap metric commonly used for summarization-style text.
- [BLEU](https://aclanthology.org/P02-1040/): n-gram precision metric originally designed for machine translation.
- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685): research context for model-based evaluation.
- [G-Eval](https://arxiv.org/abs/2303.16634): an early reference for LLM-based evaluation with structured criteria.
