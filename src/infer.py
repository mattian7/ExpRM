# infer.py is a script to generate answers for the GSM8K test set using the trained model.
import re
import os
import numpy as np
from datetime import datetime
from utils import *
import jsonlines
import torch
from role2 import Generator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


if __name__ == '__main__':

    args = parse_argunments()
    args.model = "./model/mistral7/"
    args.batch_size = 16
    print(f"loading model from {args.model}......")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map = "auto",
        torch_dtype = torch.bfloat16
    )
    print("model done")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side  = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("tokenizer done")
    
    args.data = "./datasets/gsm8k/test/test.jsonl"

    dataloader = setup_data_loader(args)

    # Generator‘s job is to generate the candidate answer with a prefix: "Let's think step by step. "
    gen = Generator(model, args.temperature, tokenizer, args)

    correct_num = 0
    total_num = 0
    print(f"batch size: {args.batch_size}")

    args.max_id = 199
    args.resume_id = 0
    output_file = "./datasets/gsm8k/test/gen/mistral7bgen.jsonl"

    for i, data in enumerate(dataloader):
        if i < args.resume_id:
            continue

        questions, GT, references = data

        candidate_answer = gen.step(questions,args)

        y_predicts = [answer_cleaning(args, answer) for answer in candidate_answer]

        position = [get_position(y_predicts[i],GT[i]) for i in range(len(y_predicts))]

        log = []
        for k in range(args.batch_size):
            if position[k] == 1:
                correct_num += 1
            tmp = {"question":questions[k], "ref":references[k], "groundtruth":GT[k], "candidate":"Let's think step by step. " + candidate_answer[k],"flag":position[k]}
            log.append(tmp)

        total_num += args.batch_size
        
        acc = correct_num / total_num * 100
        print(f"accuracy: {acc}%")

        with jsonlines.open (output_file, "a") as file:
            for l in log:
                file.write (l)        
        
        if i >= args.max_id:
            break
        