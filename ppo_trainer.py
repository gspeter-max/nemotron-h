from transformers import AutoModelForCausalLmWithValueHead , AutoTokenizer 
from trl import PPOConfig , PPOTrainer 
from datasets import load_datasets 

tokenizer = AutoTokenizer.from_pretrained( config.model_name ) 
config = PPOConfig(
        model_name = './sft_model', 
        learning_rate = 0.001,
        batch_size = 256, 
        mini_batch_size = 1, 
        gradient_accumulation_steps = 1, 
        log_with = 'wandb'
        ) 

generate_config = {
        'max_new_token' : 30, 
        'top_p' : 1, 
        'top_k': 0,
        'pad_token_ids' : tokenizer.eos_token_id
        } 

datasets = load_datasets( 'datset_name ') 
model = AutoModelForCausalLmWithValueHead.from_pretrained( config.model_name ) 

ppo_trainer = PPOTrainer( config , model , tokenizer , dataset ) # it automatically tokenizer that and you able to access this 

for epoch in ppo_trainer.epochs:
    for batch in ppo_trainer.batch:
        query = batch['query_batch']
        response =  ppo_trainer.generate( query, **generate_config )
        batch['response'] = tokenizer.decode_batch( response ) 
        
        query_response = [ q + a for q , a  in zip( batch['query_batch'] , batch['response'] ) ] 
        reward_score = reeward_model( query_respone ) 
        stats = ppo_trainer.step( reward_score , query . response ) 
        ppo_trainer.log_stats( batch, reward_score , stats ) 

