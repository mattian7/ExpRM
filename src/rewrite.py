# 使用gpt4对mistral生成的错误答案进行重写
import re
import os
import numpy as np
from datetime import datetime
from utils import *
import jsonlines
import torch
from role2 import Corrector, Incorrector, RewriterLangchain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_openai import ChatOpenAI


if __name__ == '__main__':

    os.environ["OPENAI_API_KEY"] = "api-key"
    os.environ["OPENAI_API_BASE"] = "baseurl"
    chatmodel = ChatOpenAI(model="gpt-4")

    args = parse_argunments()
    args.batch_size = 16
    
    args.data = "./datasets/gsm8k/train/gen1.jsonl"

    dataloader = setup_data_loader2(args)

    rewriter = RewriterLangchain(chatmodel, args.temperature, None, args)

    correct_num = 0
    total_num = 0
    print(f"batch size: {args.batch_size}")

    args.max_id = 199
    args.resume_id = 0
    rewritten_datafile = "./datasets/gsm8k/train/rewritten_all.jsonl"

    for i, data in enumerate(dataloader):

        questions, ca, pos, ref = data

        reref = rewriter.step(ref, ca, args)

        log = []
        for k in range(args.batch_size):
            if int(pos[k]) == 0:
                tmp = {"question":questions[k], "candidate_answer":ca[k], "ref":ref[k], "re-ref": reref[k]}
                log.append(tmp)

        with jsonlines.open (rewritten_datafile, "a") as file:
            for l in log:
                file.write (l)        
        
