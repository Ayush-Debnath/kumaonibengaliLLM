# KumaoniBengaliLLM 🏔️🗣️

A low-resource multilingual language model fine-tuned to understand and generate **Kumaoni** — an endangered Indo-Aryan dialect of Uttarakhand — alongside **Bengali**. Built using a two-stage curriculum fine-tuning approach on mT5-small with ~38K Kumaoni training examples and task-prefix tokens.

> *Preserving endangered languages through modern NLP — one token at a time.*

---

## Why This Project?

Kumaoni is spoken by over 2 million people in the Kumaon hills of Uttarakhand, India — yet it has virtually **zero presence in modern NLP research**. There is no publicly available tokenizer, no benchmark dataset, and no language model trained on Kumaoni text.

This project is an attempt to change that. By fine-tuning a multilingual foundation model on collected Kumaoni data, the goal is to demonstrate that even severely low-resource languages can benefit from modern transfer learning — and to lay groundwork for future NLP tools in the Kumaoni language community.

---

## Key Features

- **Two-stage curriculum training** — Stage 1 adapts the model to Kumaoni language patterns from raw text; Stage 2 fine-tunes on structured instruction/chat pairs
- **Task-prefix token design** — a single model handles multiple tasks (`translate_kumaoni_to_english:`, `kumaoni_chat:`, etc.) inspired directly by the mT5/T5 pretraining paradigm
- **~38K Kumaoni training examples** — collected, cleaned, and formatted into seq2seq pairs
- **Working chatbot UI** — a dark-themed conversational interface built in Google Colab with Sarvam AI API integration
- **Bengali multilingual support** — shared multilingual representations allow cross-lingual transfer between Kumaoni and Bengali

---

## Model Architecture

| Component | Details |
|---|---|
| Base model | `google/mt5-small` (300M multilingual parameters) |
| Fine-tuning strategy | Two-stage curriculum learning |
| Training data | ~38K Kumaoni seq2seq examples |
| Task framing | Text-to-text with task-prefix tokens |
| Quantization | QLoRA (4-bit) for memory-efficient fine-tuning |
| Inference backend | Sarvam AI API (chatbot UI) |

---

## Repository Structure

```
kumaonibengaliLLM/
│
├── kumaoni_stage1/              # Stage 1: Language adaptation
│   └── kumaoni_stage1.ipynb    # Base fine-tuning on raw Kumaoni corpus
│
├── kumaoni_stage2/              # Stage 2: Instruction & chat fine-tuning
│   └── kumaoni_stage2.ipynb    # Task-specific training on structured pairs
│
├── kumaoni_chat_model/          # Final chat-capable model
│   └── kumaoni_chat_mT5.ipynb  # Inference pipeline + evaluation
│
├── kumaoni_chatbot_working.ipynb  # Working chatbot UI (Colab, dark-themed)
│
├── models/                      # Saved model weights / LoRA adapters
│
├── app/                         # Serving layer (Gradio / standalone HTML)
│
├── requirements.txt
└── README.md
```

---

## Training Pipeline

### Stage 1 — Language Adaptation
The base `mT5-small` model is fine-tuned on raw Kumaoni text to develop a prior over the language's vocabulary, script patterns, and morphology. Since Kumaoni uses Devanagari script, mT5's multilingual tokenizer already has partial coverage — this stage fills in the gaps specific to the dialect.

```
Input:  kumaoni_chat: तुम कस छा?
Output: तुम कैसे हो?
```

### Stage 2 — Instruction & Chat Fine-Tuning
Building on the Stage 1 checkpoint, this stage trains on structured `(instruction, response)` pairs across multiple task types using prefix tokens:

| Prefix Token | Task |
|---|---|
| `kumaoni_chat:` | Conversational response in Kumaoni |
| `translate_kumaoni_to_hindi:` | Kumaoni → Hindi translation |
| `translate_kumaoni_to_english:` | Kumaoni → English translation |
| `kumaoni_qa:` | Question answering in Kumaoni |

This design lets a single model handle diverse capabilities without task-specific heads — exactly how the original T5 paper intended seq2seq models to work.

---

## Chatbot Demo

The `kumaoni_chatbot_working.ipynb` notebook contains a fully working chat interface with:
- Dark-themed HTML/CSS UI rendered inside Google Colab
- Multi-turn conversation with history tracking
- `<think>` tag stripping for clean output
- Token limit handling for long conversations
- Sarvam AI API as the inference backend

---

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
```
transformers>=4.35.0
datasets
peft
torch
accelerate
bitsandbytes
sentencepiece
```

### Run the Chatbot (Colab)

1. Open `kumaoni_chatbot_working.ipynb` in Google Colab
2. Set your Sarvam AI API key in the secrets panel
3. Run all cells — the chat UI will render inline

### Fine-tune from Scratch

Open `kumaoni_stage1/kumaoni_stage1.ipynb` and follow the cells:

```python
# Load base model
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

# Load and format your Kumaoni dataset
# Add task prefix tokens
# Train with Seq2SeqTrainer
```

Then run `kumaoni_stage2/kumaoni_stage2.ipynb` on the Stage 1 checkpoint.

---

## The Low-Resource Challenge

Training NLP models on Kumaoni presents unique difficulties:

**Data scarcity** — There is no large existing corpus. Training data was collected and curated manually, resulting in ~38K examples — tiny by modern standards but sufficient for fine-tuning a multilingual base model.

**Tokenizer coverage** — mT5's SentencePiece tokenizer has incomplete Kumaoni coverage since the language was not in its pretraining data. This means Kumaoni words are often split into many subword tokens, increasing sequence lengths and training difficulty.

**No evaluation benchmark** — There is no established BLEU or accuracy benchmark for Kumaoni NLP tasks. Model evaluation was done through human inspection of generated outputs.

**Dialect variation** — Kumaoni has significant dialectal variation across districts (Almora, Nainital, Pithoragarh, etc.). The current model is trained primarily on a central dialect.

---

## Results (Qualitative)

The model shows encouraging results for a first iteration on such a low-resource language:

- Coherent conversational responses in Kumaoni for common greetings, daily phrases, and simple questions
- Reasonable Kumaoni → Hindi translation for short sentences
- Cross-lingual transfer visible: Bengali queries occasionally resolve correctly without explicit training

A quantitative benchmark with BLEU scores is planned for a future release as a proper evaluation dataset is assembled.

---

## Future Work

- [ ] Expand training corpus beyond 38K examples with web-scraped and community-sourced data
- [ ] Build and release a public Kumaoni NLP benchmark dataset
- [ ] Fine-tune on a larger base model (mT5-base or Sarvam-1)
- [ ] Add support for more Kumaoni dialects
- [ ] Publish the dataset and model weights to Hugging Face Hub
- [ ] Write and submit a short paper to a low-resource NLP workshop (e.g., AmericasNLP, WILDRE)

---

## Motivation

This project is deeply personal. Kumaoni is a living cultural heritage — spoken in the hill villages of Uttarakhand but rapidly declining among younger generations as Hindi and English dominate. NLP tools that understand and generate Kumaoni can help in:

- Digitizing and preserving oral literature and folk songs
- Building voice assistants accessible to elderly Kumaoni speakers
- Educational tools for children learning their native dialect
- Machine translation for government documents into local languages

---

## Author

**Ayush Debnath**  
B.Tech Computer Science Engineering — Graphic Era Hill University  
Research interests: Low-resource NLP, Multilingual LLMs, Language Preservation  
[GitHub](https://github.com/Ayush-Debnath) · debnathayush48@gmail.com

---



## Acknowledgements

- [Google mT5](https://github.com/google-research/multilingual-t5) — the multilingual foundation model this work builds on
- [Hugging Face Transformers](https://github.com/huggingface/transformers) — training and inference framework
- [Sarvam AI](https://www.sarvam.ai/) — Indian language model API used in the chatbot interface
- The Kumaoni-speaking community, whose language and stories are worth preserving