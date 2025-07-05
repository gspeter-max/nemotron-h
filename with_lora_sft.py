# ! huggingface-cli login
# ! pip install bitsandbytes
# ! pip install trl
# ! pip install -U transformers
# ! pip install -U datasets

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True
)

dataset = load_dataset('bjoernp/ultrachat_de', split = 'train')
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    quantization_config = quantization_config,
    device_map = 'auto'
)
processor = AutoProcessor.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')
tokenizer.pad_token = tokenizer.eos_token

print(processor.__class__.__name__)
print(processor.chat_template)
print(model.__class__.__name__)
print(tokenizer.__class__.__name__)
print(tokenizer.chat_template)

dataset = dataset.rename_columns(
    {
        'response' : 'completion'
    }
)


sft_config= SFTConfig(
        do_train = True,
        per_device_train_batch_size = 2,
        learning_rate = 0.0001,
        weight_decay = 0.002,
        num_train_epochs = 12,
        warmup_steps = 23
    )

# for layers in model.named_modules():
#     print(layers)

lora_config = LoraConfig(
        r = 12,
        lora_alpha = 23,
        lora_dropout = 0.002,
        target_modules = 'all-linear'
    )

sft_trainer = SFTTrainer(
        model = model,
        processing_class = processor,
        args = sft_config,
        peft_config = lora_config,
        train_dataset = dataset
        )

sft_trainer.train()
