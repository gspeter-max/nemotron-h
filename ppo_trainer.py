
from transformers import AutoModelForSequenceClassification , AutoTokenizer , GenerationConfig , AutoModelForCausalLM
from trl import PPOConfig , PPOTrainer, AutoModelForCausalLMWithValueHead 
import trl 
from datasets import load_dataset 
import wandb 

''' 
wandb.init(
    project = "huggingface", 
    id = "" , 
    resume = 'must'
    )
    ''' 

tokenizer = AutoTokenizer.from_pretrained('gpt2') 
tokenizer.pad_token = tokenizer.eos_token
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
        batch_size = 2, 
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


datasets = load_dataset('trl-internal-testing/descriptiveness-sentiment-trl-style')
print(datasets)
def batch_tokenization(example):
    data = tokenizer(example['prompt'], truncation= True , padding = 'max_length')
    return data 

datasets = datasets['descriptiveness'].map(
    batch_tokenization, 
    batched = True , 
    remove_columns = datasets['descriptiveness'].column_names
    )

model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')
ref_model = AutoModelForCausalLM.from_pretrained('/content/drive/MyDrive/nemotron-h/checkpoint-1636' )
print(ref_model.__class__.__name__)
reward_model = AutoModelForSequenceClassification.from_pretrained('gpt2') 
print(reward_model.__class__.__name__)
model.generation_config = generation_config

ppo_trainer = PPOTrainer(
        args = config, 
        processing_class = tokenizer,
        model = model, 
        ref_model = ref_model, 
        train_dataset = datasets,
        reward_model = reward_model, 
        value_model = reward_model

    ) # it automatically tokenizer that and you able to access this
# for epoch in range(ppo_trainer.config.num_train_epochs):
#     for batch in ppo_trainer.batch:
#         query = batch['query_batch']
#         # print('------------------------------------------------------------------------')
#         response =  ppo_trainer.generate(query)        
#         batch['response'] = tokenizer.decode_batch( response ) 

#         query_response = [ q + a for q , a  in zip( batch['query_batch'] , batch['response'] ) ] 
#         reward_score = reward_model(query_response) 
#         stats = ppo_trainer.step( reward_score , query . response ) 
#         ppo_trainer.log_stats( batch, reward_score , stats ) 

ppo_trainer.train()

