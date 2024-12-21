"""
wandb login
tmux set mouse on
accelerate launch --config_file ./config.yaml ./src/ppo_rewrite.py
给模型一个自己生成的错误答案和一个Chatgpt rewrite后的正确答案，使用llm判断哪个是正确答案, 判断对了reward为+1, 判断错了是-1, 并使用ppo训练
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available

from utils import *
from role2 import Judger
import jsonlines
import gc
import os
from langchain_openai import ChatOpenAI


tqdm.pandas()


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="./model/mistral7b/", metadata={"help": "the model name"})
    seed: int=field(default=0, metadata={"help": "random seed"})
    use_seq2seq: bool = field(default=False, metadata={"help": "whether to use seq2seq"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "optimizer learning rate"})
    batch_size: int = field(default=8, metadata={"help": "batch size of sampling data from dataset"})
    trainer_batch_size: int = field(default=32, metadata={"help": "batch size of ppo training process(per gpu)"})
    trainer_gradient_accumulation_steps: int = field(default=4, metadata={"help": "gradient accumulation steps of ppo training process"})
    trainer_mini_batch_size: int = field(default=2, metadata={"help": "mini batch size of ppo training process(per gpu)"})
    trainer_ppo_epochs: int = field(default=4,metadata={"help": "how many times dose one data from repaly buffer are used"})
    temperature: float = field(default=0.7, metadata={"help": "Temperature when generating"})
    max_step : int = field(default=1, metadata={"help": "max step of iteration"})
    epochs : int = field(default=2, metadata={"help": "epochs of iteration"})

    # LoraConfig
    use_peft: bool = field(default=True, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})

    #dataset
    max_num_worker: int = field(default=0, metadata={"help": "num of workers tyo read data"})
    data: str = field(default="./datasets/gsm8k/train/rewritten_final.jsonl", metadata={"help": "data path"})
    distributed: bool = field(default=True, metadata={"help": "whether to use distributed training"})


parser = HfArgumentParser((ScriptArguments))
args = parser.parse_args_into_dataclasses()[0]

ppo_config = PPOConfig(
    model_name=args.model_name,
    learning_rate=args.learning_rate,
    batch_size=args.trainer_batch_size,
    seed = args.seed,
    mini_batch_size = args.trainer_mini_batch_size,
    gradient_accumulation_steps = args.trainer_gradient_accumulation_steps,
    kl_penalty = "full",
    ppo_epochs = args.trainer_ppo_epochs,
    log_with = "wandb",
    use_score_scaling = True,
    use_score_norm = True,
)

trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead


set_seed(ppo_config.seed)

if not args.use_peft:
    ref_model = trl_model_class.from_pretrained(ppo_config.model_name, trust_remote_code=args.trust_remote_code)
    device_map = None
    peft_config = None
else:
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
    ref_model = None
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}

model = trl_model_class.from_pretrained(
    ppo_config.model_name,
    trust_remote_code=args.trust_remote_code,
    device_map=device_map,
    peft_config=peft_config,
    torch_dtype = torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name,use_fast=False)
tokenizer.pad_token_id = tokenizer.eos_token_id


ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer)

dataloader = setup_data_loader3(args)
dataloader = ppo_trainer.accelerator.prepare(dataloader)

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    if is_xpu_available():
        device = "xpu:0"
    elif is_npu_available():
        device = "npu:0"
    else:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

generation_kwargs = {
    "repetition_penalty":1.3, 
    "temperature": args.temperature,
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 400,
}

judger = Judger(args.temperature, tokenizer, args)

invalid_num = 0
win_num_correct = 0
win_num_wrong = 0
total_num_correct = 0
total_num_wrong = 0

judge_file = "./datasets/gsm8k/rewrite/4.jsonl"

peft_model_save_root = "./model/ppo_rewrite4/"
merge_model_save_path = "./model/rewrite4/"
ITER_NUM = 0
STEP_NUM = 0

for epoch_ids in range(args.epochs):

    query_replay_buffer = []
    response_replay_buffer = []
    reward_replay_buffer = []

    FLAG_add_zero = 0
    for _idx, batch in tqdm(enumerate(dataloader)):
        query = []
        response = []
        
        questions, ca, re_ref = batch
        
        with torch.no_grad():
            query4j, query_tensors4j, right_answer_pos = judger.tokenize(questions,re_ref, ca, args) 
            response_tensors4j = ppo_trainer.generate(query_tensors4j, return_prompt=False, **generation_kwargs)
            judges = tokenizer.batch_decode(response_tensors4j)

            reward = np.array(clean_judges(judges,right_answer_pos))
        
            jlog = []
            for k in range(args.batch_size):
                tmp = {"question": questions[k], "answer":ca[k], "re-ref": re_ref[k], "reward":str(reward[k]), "judge":judges[k],"pos":str(right_answer_pos[k])}
                jlog.append(tmp)
                if int(reward[k]) ==1:
                    total_num_correct += 1
                else:
                    total_num_wrong += 1
                
            with jsonlines.open (judge_file, "a") as file:
                for j in jlog:
                    file.write (j)

            for i in range(len(reward)):
                query_replay_buffer.append(query_tensors4j[i])
                response_replay_buffer.append(response_tensors4j[i])
                reward_replay_buffer.append(torch.tensor(reward[i]))

            #监控胜率
            
            win_rate = total_num_correct / (total_num_correct+total_num_wrong) * 100
                
            if ppo_trainer.accelerator.is_local_main_process:
                print(f"win-rate: {win_rate}%\n")

        gc.collect()  # 执行垃圾回收
        torch.cuda.empty_cache()

        #如果经验回放池够了256条数据，则进行ppo训练
        if len(query_replay_buffer) >= ppo_config.batch_size:
            bs = ppo_config.batch_size
            query_tensors = query_replay_buffer[:bs]
            response_tensors = response_replay_buffer[:bs]
            rewards = reward_replay_buffer[:bs]

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            del query_tensors, response_tensors, rewards
            gc.collect() 
            torch.cuda.empty_cache()            
            
            query_replay_buffer = []
            response_replay_buffer = []
            reward_replay_buffer = []

            ITER_NUM += ppo_config.batch_size/(ppo_config.backward_batch_size)

            if ppo_trainer.accelerator.is_local_main_process:
                print(f"iter_num:{ITER_NUM}")

            if ITER_NUM % 12 == 0:
                peft_model_save_pth = peft_model_save_root + str(epoch_ids) + "epoch_"+ str(ITER_NUM) + "/"
                ppo_trainer.save_pretrained(peft_model_save_pth)  

            query_tensors = None
            response_tensors = None
            rewards = None

            gc.collect()  # 执行垃圾回收
            torch.cuda.empty_cache()

    peft_model_save_pth = peft_model_save_root + str(epoch_ids) + "_epoch/"
    ppo_trainer.save_pretrained(peft_model_save_pth)

    query_tensors = None
    response_tensors = None
    rewards = None

    gc.collect()  # 执行垃圾回收
    torch.cuda.empty_cache()

del ppo_trainer, model  # 删除模型和训练器的引用，释放内存
gc.collect()  # 执行垃圾回收
torch.cuda.empty_cache()  # 清空CUDA缓存，释放GPU内存

base_model = trl_model_class.from_pretrained(
    args.model_name,  
    return_dict=True, 
    torch_dtype=torch.bfloat16,  
)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # 重新加载分词器

model = PeftModel.from_pretrained(base_model, peft_model_save_pth)  # 从基础模型和"final_checkpoint"目录加载PEFT模型
model = model.merge_and_unload()  # 合并基础模型和适配器，卸载不必要的部分

model.save_pretrained(merge_model_save_path)  # 将合并后的模型保存到new_model指定的目录
tokenizer.save_pretrained(merge_model_save_path)  # 将分词器保存到new_model指定的目录

print("saved....")