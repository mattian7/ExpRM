# Merge the base model and the PEFT model to a new model.
import re
import os
import numpy as np
from datetime import datetime
from utils import *
import jsonlines
import torch
from role3 import TruthTeller, Lier, Judger,JudgerLangchain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_openai import ChatOpenAI


def fun(a,b,*args,**kwargs):
    print(f"a is {a}, b is {b}")
    print(f"args is {args}")
    print(f"kwargs is {kwargs}")

if __name__ == '__main__':
    
    args = parse_argunments()
    
    model_name = "./model/mistral7b/"
    peft_model_save_pth = "./model/ppo_rewrite3/1_epoch/"
    merge_model_save_path = "./model/rewrite3_1epoch/"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,  
        return_dict=True, 
        torch_dtype=torch.bfloat16,  
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # 重新加载分词器
    
    model = PeftModel.from_pretrained(base_model, peft_model_save_pth)  # 从基础模型和"final_checkpoint"目录加载PEFT模型
    model = model.merge_and_unload()  # 合并基础模型和适配器，卸载不必要的部分

    model.save_pretrained(merge_model_save_path)  # 将合并后的模型保存到new_model指定的目录
    tokenizer.save_pretrained(merge_model_save_path)  # 将分词器保存到new_model指定的目录

    print("saved....")
    '''
    model_judge = PeftModel.from_pretrained(base_model, peft_model_save_pth_judge)  # 从基础模型和"final_checkpoint"目录加载PEFT模型
    model_judge = model_judge.merge_and_unload()  # 合并基础模型和适配器，卸载不必要的部分

    model_judge.save_pretrained(merge_model_save_path_judge)  # 将合并后的模型保存到new_model指定的目录
    tokenizer.save_pretrained(merge_model_save_path_judge)  # 将分词器保存到new_model指定的目录

    print("judge saved....")
    

    os.environ["OPENAI_API_KEY"] = "sk-RMUohnSysYZijXFFDe11F0D8E5194a8fAf129d3fA875D0Fa"
    os.environ["OPENAI_API_BASE"] = "http://47.236.144.103/v1"
    chatmodel = ChatOpenAI(model="gpt-4")
    judger = JudgerLangchain(chatmodel, 0, None, args)
    
    questions = ["what is the value of one plus one?","what is the value of one plus three?"]
    candidates = ["The answer is 2","The answer is 3"]
    history = ["Oponent: The answer can't be 2 because 1 is bigger than 2.\n\nProponent: Are u kidding me? one is the half of 2, so the answer is right.","Oponent: The answer can't be 3 because three plus one must bigger than one.\n\nProponent: I don't think so, i guess this answer is right"]
    judgements = judger.step(questions, candidates, history, args) 
    print(judgements)
    print([clean2(j) for j in judgements])
    ''' 
   