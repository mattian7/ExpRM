# python ./src/reward_test2.py
# 在测试集上计算模型对于给定问题和(由base_model生成的)候选答案的分类准确率（acc_p, acc_o）
import re
import os
import numpy as np
from datetime import datetime
from utils import *
import jsonlines
import torch
from role2 import Detective
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


if __name__ == '__main__':

    args = parse_argunments()
    args.model = "./model/rewrite3_1epoch/"
    args.batch_size = 16
    print(f"loading model from {args.model}......")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map = "auto",
        torch_dtype = torch.bfloat16,
        trust_remote_code = True
    )
    print("model done")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side  = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("tokenizer done")
    
    #mistral7bgen.jsonl中包含了问题questions、由mistral7b生成的候选答案ca、pos=1表示该ca是正确答案，pos=0表示该ca是错误答案
    args.data = "./datasets/gsm8k/test/mistral7bgen.jsonl"

    dataloader = setup_data_loader2(args)

    # Detective‘s job is to judge the candidate answer is correct or not
    detect = Detective(model, args.temperature, tokenizer, args)

    correct_num = 0
    correct_num_p = 0
    correct_num_o = 0
    total_num = 0
    total_num_p = 0
    total_num_o = 0
    print(f"batch size: {args.batch_size}")

    args.max_id = 19

    eval_outputfile = "./datasets/gsm8k/test/judger-output.jsonl"

    for i, data in enumerate(dataloader): 
        if i < args.resume_id:
            continue

        questions, ca, pos, ref = data

        decision = detect.step(questions,ca,args)
        #print(decision)
        y_predicts = [clean_model_decision(pre) for pre in decision]
        

        log = []
        for k in range(args.batch_size):
            if y_predicts[k] == int(pos[k]):
                correct_num += 1
            
            if int(pos[k])== 1:
                total_num_p += 1
                if y_predicts[k] == 1:
                    correct_num_p += 1

            if int(pos[k])== 0:
                total_num_o += 1
                if y_predicts[k] == 0:
                    correct_num_o += 1
            tmp = {"question":questions[k],"candidate":ca[k],"decision":decision[k],"flag":str(y_predicts[k]),"pos":str(pos[k])}
            log.append(tmp)

        total_num += args.batch_size
        
        acc = correct_num / total_num * 100
        acc_p = correct_num_p / total_num_p * 100
        acc_o = correct_num_o / total_num_o * 100

        print(f"accuracy: {acc}%; acc_correct: {acc_p}%; acc_incorrect: {acc_o}%")

        
        with jsonlines.open (eval_outputfile, "a") as file:
            for l in log:
                file.write (l)        
        
        if i >= args.max_id:
            break