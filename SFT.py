from transformers import (
     AutoModelForCausalLM ,
     AutoTokenizer ,
     Trainer,
     TrainingArguments
    )

import wandb
wandb.init(
    project = 'huggingface', 
    id = 'ngpez7di', 
    resume = 'must'
)
from datasets import load_dataset
# from trl import SFT_Trainer 

dataset = load_dataset('domenicrosati/TruthfulQA') 

model_name = "/content/drive/MyDrive/nemotron-h/checkpoint-818"
model = AutoModelForCausalLM.from_pretrained ( model_name )
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token  = tokenizer.eos_token 

print(dataset)
def data_tokenizer(example):
    inputs = [f'question : {question} Best Answer : {best_answer}' for question , best_answer in zip( example['Question'], example['Best Answer'])] 
    model_inputs = tokenizer( inputs , truncation = True, return_tensors = 'pt', padding = 'max_length' )
    model_inputs['labels'] = model_inputs['input_ids'] 

    return model_inputs 

batched_datasets = dataset.map( data_tokenizer , batched = True, remove_columns = dataset['train'].column_names ) 
traning_arg = TrainingArguments(
        output_dir = '/content/drive/MyDrive/nemotron-h', # where  you like to store ( like drive )
        num_train_epochs = 20,

        per_device_train_batch_size = 20,
        per_device_eval_batch_size = 20, 
        warmup_steps = 100,
        save_strategy = 'epoch',
        save_total_limit = 3,
        report_to = 'wandb',
        logging_steps = 10,
        weight_decay = 0.003, 
        logging_dir = './log_output'
        ) 

trainer = Trainer(
            train_dataset = batched_datasets['train'], 
            model = model,
            args = traning_arg 
        )

trainer.train(resume_from_checkpoint = True) 


