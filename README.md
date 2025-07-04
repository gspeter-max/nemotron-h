# Nemotron-H: Fine-Tuning and PPO Training Framework 🤖

This repository contains an end-to-end framework for training and aligning large language models using two critical stages:

* **SFT (Supervised Fine-Tuning)**
* **PPO (Proximal Policy Optimization)**

It is designed to enable powerful and efficient reward-aligned fine-tuning using modern deep RL techniques.

---

## 🎓 Features

### ✅ Supervised Fine-Tuning (SFT)

Train your base language model using high-quality human-annotated datasets with cross-entropy loss.

* Dataset: [`domenicrosati/TruthfulQA`](https://huggingface.co/datasets/domenicrosati/TruthfulQA)
* Model checkpoint used: `/content/drive/MyDrive/nemotron-h/checkpoint-818`
* Tokenizer: GPT-2 with EOS token as pad token

### ⚖️ Reward Modeling

Train a reward model using datasets with human preference annotations to distinguish between good and bad responses.

* Dataset: [`argilla/dpo-mix-7k`](https://huggingface.co/datasets/argilla/dpo-mix-7k)
* Reward model: `distilbert-base-uncased` for pairwise scoring
* Contrastive loss using chosen vs rejected examples

### 🏆 PPO Trainer

Fine-tune the language model using reinforcement learning based on feedback from the reward model.

* Dataset: [`allura-org/instruct-ppo-mix-20k`](https://huggingface.co/datasets/allura-org/instruct-ppo-mix-20k)
* Base + reference model: `/content/drive/MyDrive/nemotron-h/checkpoint-1636`
* Reward model used: Custom checkpoint from `reward_model/checkpoint-1125`
* Built on `trl` PPOTrainer

---

## 🔧 Project Structure

```bash
.
├── SFT.py               # Script for supervised fine-tuning
├── reward_model.py      # Reward model architecture and training
├── ppo_trainer.py       # PPO training loop using reward model
├── wandb.env            # WANDB API key (not for public sharing)
├── .vscode/             # VSCode settings (optional)
```

---

## 💡 Quick Start

### 1. Setup

```bash
pip install -r requirements.txt
```

### 2. Train Supervised Model

```bash
python SFT.py
```

### 3. Train Reward Model

```bash
python reward_model.py
```

### 4. Run PPO Fine-Tuning

```bash
python ppo_trainer.py
```

---

## 🌟 Highlights

* End-to-end alignment framework (SFT + RM + PPO)
* Real open-source alignment datasets used
* Modular and hackable architecture
* Easy integration with WANDB for live monitoring
* GPT-style tokenizer and transformer models

---

## 🚀 Future Work

* Add `requirements.txt`
* Add evaluation scripts
* LoRA/QLoRA support
* Add more datasets (e.g., Anthropic HH, OpenAssistant)
* HuggingFace Spaces UI for inference demo

---

## 🚀 Author

[**@gspeter-max**](https://github.com/gspeter-max)

This repo is actively evolving. Contributions, ideas, and pull requests are welcome!

