from transformers import AutoModelForSequenceClassification , AutoTokenizer , GenerationConfig
from trl import PPOConfig , PPOTrainer, AutoModelForCausalLMWithValueHead 
from datasets import load_dataset 
import wandb 

''' 
wandb.init(
    project = "huggingface", 
    id = "" , 
    resume = 'must'
    )
    ''' 

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased') 
tokenizer.pad_tokens = tokenizer.eos_token
config = PPOConfig( 
        learning_rate = 0.001,
        do_train = True, 
        per_device_train_batch_size = 3, 
        per_device_eval_batch_size = 3, 
        weight_decay = 0.001, 
        num_train_epochs = 12, 
        warmup_steps = 30, 
        logging_dir = './ppo_training_config', 
        save_strategy = 'epoch', 
        logging_steps = 10,
        batch_size = 256, 
        mini_batch_size = 1, 
        gradient_accumulation_steps = 1, 
        ) 

generation_config = GenerationConfig.from_pretrained(
  pretrained_model_name = '/content/drive/MyDrive/nemotron-h/checkpoint-1636',
    max_new_tokens=100,
    min_length=2,
    top_k=0.0,
    top_p=1.0,
    do_sample=True,
    pad_token_id=50256,  # Often set to the EOS token ID
    eos_token_id=50256

)


datasets = load_dataset('allura-org/instruct-ppo-mix-20k') 
model = AutoModelForCausalLMWithValueHead.from_pretrained('/content/drive/MyDrive/nemotron-h/checkpoint-1636' )
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained('/content/drive/MyDrive/nemotron-h/checkpoint-1636' )
reward_model = AutoModelForSequenceClassification.from_pretrained('/content/drive/MyDrive/reward_model/checkpoint-1125') 
model.generation_config = generation_config
print('---------------------------------------------dgfgdf----------------------------------')
ppo_trainer = PPOTrainer(
        args = config, 
        processing_class = tokenizer,
        model = model, 
        ref_model = ref_model, 
        train_dataset= datasets, 
        reward_model = reward_model, 
        value_model = reward_model

    ) # it automatically tokenizer that and you able to access this 
print('--------------------------------------------------------------sdgfs-----------------')
print(vars(ppo_trainer))
for epoch in range(ppo_trainer.args.num_train_epochs):
    for batch in ppo_trainer.dataloader:
        query = batch['query_batch']
        # print('------------------------------------------------------------------------')
        response =  ppo_trainer.generate(query)
        
        batch['response'] = tokenizer.decode_batch( response ) 

        
        query_response = [ q + a for q , a  in zip( batch['query_batch'] , batch['response'] ) ] 
        reward_score = reeward_model( query_respone ) 
        stats = ppo_trainer.step( reward_score , query . response ) 
        ppo_trainer.log_stats( batch, reward_score , stats ) 


