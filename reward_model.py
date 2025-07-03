
'''
1. human  prefercen datasets
2. model to train like this is best response and that is bad
3. we are combine out and show --> { 'questions : {question}  best_answer : {best_answer}  bad_answer : { bad_answer } in line str '
                so the model is easily find out the patterns like bad_answer --> ooo that is bad asign that '
'''

from transformers import AutoModelForSequenceClassification , AutoTokenizer , Trainer , TrainingArguments
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional

import torch.nn.functional as F

# Load a dataset of human preferences
dataset = load_dataset("argilla/dpo-mix-7k")
reward_model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained ( reward_model_name )
reward_mdoel = AutoModelForSequenceClassification.from_pretrained ( reward_model_name )
print(dataset)
def tokenizer_func( example ):
    
    chosen = []
    rejected = []

    for chosen_example, reject_example in zip( example['chosen'], example['rejected']):
        chosen_example = f"question : {chosen_example[0]['content']} chosen_answer : {chosen_example[1]['content']}"
        reject_example = f"question : {reject_example[0]['content']} chosen_answer : {reject_example[1]['content']}"

        chosen_tokenizer_output  = tokenizer( chosen_example , truncation = True , padding = 'max_length')
        chosen.append(chosen_tokenizer_output)
        # new_example['chosen_input_ids'].append( chosen_tokenizer_output.input_ids )
        # new_example['chosen_attention_mask'].append( chosen_tokenizer_output.attention_mask )

        reject_tokenizer_output = tokenizer( reject_example , truncation = True , padding = 'max_length' )
        rejected.append(reject_tokenizer_output)
        # new_example['reject_input_ids'].append( reject_tokenizer_output.input_ids )
        # new_example['reject_attention_mask'].append( reject_tokenizer_output.attention_mask )

    return {'chosen' : chosen , 'rejected' : rejected }
tokenized_dataset = dataset.map(tokenizer_func , batched = True , remove_columns = dataset['train'].column_names)
import torch

class RewardTrainer( Trainer ):
    def compute_loss(self,model,inputs,num_items_in_batch,output_features = False ):
        chosen_reward = model(
            inputs['chosen_input_ids'],
            attention_mask = inputs['chosen_attention_mask']
        )[0]
        rejected_reward = model(
            inputs['rejected_input_ids'],
            attention_mask = inputs['rejected_attention_mask']
        )[0]
        loss = torch.mean(-F.logsigmoid( chosen_reward - rejected_reward), -1 )

        return (loss, {'chosen_reward' : chosen_reward , 'rejected_reward' : rejected_reward }) if output_features else loss

@dataclass
class RewardDataCollatorWithPadding:

    tokenizer : AutoTokenizer
    padding : bool = True
    max_length : Optional[int] = None

    def __call__( self, features ):
        chosen_features  = []
        rejected_features = []
        for feature in features:
            chosen_features.append(
                {
                    'input_ids' : feature['chosen'].get('input_ids'),
                    'attention_mask' : feature['chosen'].get('attention_mask')
                }
            )
            rejected_features.append(
                {
                    'input_ids': feature['rejected'].get("input_ids"),
                    'attention_mask' : feature['rejected'].get("attention_mask")
                }
            )

            chosen_padded = self.tokenizer.pad(
                chosen_features,
                padding = 'max_length',
                return_tensors = 'pt'
            )
            rejected_padded = self.tokenizer.pad(
                rejected_features,
                padding = 'max_length',
                return_tensors = 'pt'
            )

            return {
                'chosen_input_ids' : chosen_padded['input_ids'] ,
                'chosen_attention_mask' : chosen_padded['attention_mask'],
                'rejected_input_ids' : rejected_padded['input_ids'],
                'rejected_attention_mask' : rejected_padded['attention_mask']
            }

arg = TrainingArguments(
        output_dir = './output',
        num_train_epochs= 1,
        per_device_train_batch_size = 60 ,
        per_device_eval_batch_size = 60,
        warmup_steps = 129,
        weight_decay = 0.002,
        remove_unused_columns=False
    )
print(tokenized_dataset['train'])
trainer = RewardTrainer(
        model = reward_mdoel,
        train_dataset = tokenized_dataset['train'],
        data_collator = RewardDataCollatorWithPadding( tokenizer = tokenizer ),
        args = arg
        )

trainer.train()