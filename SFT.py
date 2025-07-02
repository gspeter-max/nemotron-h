from transformers import (
     AutoModelForCausalLM ,
     AutoTokenizer ,
     Trainer,
     TrainingArguments
    )
from datasets import load_dataset
# from trl import SFT_Trainer 

dataset = load_dataset('domenicrosati/TruthfulQA') 

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained ( model_name )
tokenizer = AutoTokenizer.from_pretrained( model_name )
tokenizer.pad_token  = tokenizer.eos_token 


def data_tokenizer(example):
    inputs = [f'question : {question} Best Answer : {best_answer}' for question , best_answer in zip( example['question'], example['best_answer'])] 
    model_inputs = tokenizer( inputs , truncation = True, return_tensors = 'pt', padding = 'max_length' )
    model_inputs['labels'] = model_inputs['input_ids'] 

    return model_inputs 

batched_datasets = dataset.map( data_tokenizer , batched = True, remove_columns = dataset['train'].column_names ) 
traning_arg = TrainingArguments(
        output_dir = './output_model',
        num_train_epochs = 1, 
        per_device_train_batch_size = 20,
        per_device_eval_batch_size = 20, 
        warmup_steps = 600, 
        weight_decay = 0.003 , 
        logging_dir = './log_output' 
        ) 

trainer = Trainer(
            train_dataset = batched_datasets['train'], 
            model = model,
            args = traning_arg 
        )

trainer.train() 



