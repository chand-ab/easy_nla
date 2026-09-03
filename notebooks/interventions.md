# Using a published NLA: interventions and paraphrase

Load a released Gemma-3-12B NLA and intervene on the AV's text. No training, no
vLLM, no datagen.

## Install

```bash
git clone https://github.com/chand-ab/easy_nla.git && cd easy_nla
python -m venv .venv && source .venv/bin/activate   
pip install -e .          # torch, transformers==4.57.1, peft, safetensors, pyyaml
hf auth login             # google/gemma-3-12b-it is gated
```

## Download

Available repos:
`achand45/gemma-3-12b-it-nla-L{24,32,40,47}`

```bash
hf download achand45/gemma-3-12b-it-nla-L32 --local-dir ./nla-L32 \
  --include 'rl_vllm/iter_000400/*' 'rl_vllm/critic_latest/*' 'rl_vllm/nla_meta.yaml'
```

`iter_000400` = RL-final AV LoRA; `critic_latest` = matching AR (~20 GB). Base is
`google/gemma-3-12b-it`; base + AR fit one 80 GB card, `--ar-device` splits them.

## Run

```bash
python notebooks/nla_intervene.py --nla-dir ./nla-L32/rl_vllm
```
