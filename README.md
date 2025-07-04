# Nemotron-H: Fine-Tuning and PPO Training Framework 🤖

This repository contains an end-to-end framework for training and aligning large language models using two critical stages:

* **SFT (Supervised Fine-Tuning)**
* **PPO (Proximal Policy Optimization)**

It is designed to enable powerful and efficient reward-aligned fine-tuning using modern deep RL techniques.

---

## 🎓 Features

### ✅ Supervised Fine-Tuning (SFT)

Train your base language model using high-quality human-annotated datasets with cross-entropy loss.

* Hugging Face Transformers compatible
* Simple plug-and-play with custom datasets
* Clear modular structure

### ⚖️ Reward Modeling

Define and train your custom reward model.

* Architecture-agnostic reward model support
* Seamless integration with PPO trainer

### 🏆 PPO Trainer

Train your model with reinforcement learning using reward signal from the reward model.

* Built with `trl` PPOTrainer
* Supports gradient accumulation, KL penalty, etc.
* Includes logging via Weights & Biases

---

## 🔧 Project Structure

```bash
.
├── SFT.py               # Script for supervised fine-tuning
├── ppo_trainer.py       # PPO training loop using reward model
├── reward_model.py      # Reward model architecture and training
├── wandb.env            # WANDB API keys or config file
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

* Modular design (easily extendable)
* Hugging Face and TRL compatible
* Plug in your own reward model
* Easy WANDB integration
* Full pipeline in one repo

---

## 🚀 Future Work

* Add `requirements.txt`
* Add example datasets
* Add LoRA/QLoRA support
* Add evaluation metrics

---

## 🚀 Author

**[@gspeter-max](https://github.com/gspeter-max)**

This repo is actively evolving. Contributions, ideas, and pull requests are welcome!

