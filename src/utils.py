# 各种其他函数，比较乱，有些有重复，后续整理
import argparse
import multiprocessing
import torch
import numpy as np
import re
import random
import json
from statistics import mean
from torch.utils.data import Dataset

def parse_argunments():
    parser = argparse.ArgumentParser()

    parser.add_argument ("--few_shot_path", type=str, default="./demonstration/few-shot.json")
    parser.add_argument ("--model", type=str, default="./model/mistral7b/")
    parser.add_argument ("--temperature", type=float, default=0.7)
    parser.add_argument ("--seed", type=int, default=1, help="set random seed")
    parser.add_argument ("--batch_size", type=int, default=8)
    parser.add_argument ("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")
    parser.add_argument ("--resume_id", type=int, default=0, help="from which batch (sorting from 0) to continue running")
    parser.add_argument ("--max_id", type=int, default=50, help="to which batch (sorting from 0) to stop running")
    parser.add_argument ("--iter_time", type=int, default=1)
    parser.add_argument ("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument ("--method", type=str, default="zero-shot", choices=["zero-shot", "one-shot", "few-shot"])
    parser.add_argument ("--lr", type=float, default=1e-5)
    parser.add_argument ("--cpt_save_root", type=str, default="./model/mistral-7b-iter")
    parser.add_argument ("--data", type=str, default="./datasets/gsm8k/train/train.jsonl")

    args = parser.parse_args ()

    return args

def fix_seed(seed):
    random.seed (seed)
    np.random.seed (seed)
    torch.manual_seed (seed)
    torch.cuda.manual_seed_all (seed)


def data_reader(args):
    questions = []
    answers = []
    ref = []
    decoder = json.JSONDecoder ()


    path = args.data
    with open (path) as f:
        lines = f.readlines ()
        for line in lines:
            json_res = decoder.raw_decode (line)[0]
            questions.append (json_res["question"].strip ())
            answers.append (json_res["answer"].split ("#### ")[-1].replace(",",""))
            ref.append (json_res["answer"].split ("#### ")[0].strip())

    q_len_list = []
    for q in questions:
        q_len_list.append (len (q.split (" ")))
    q_len_mean = mean (q_len_list)
    
    print ("dataset : {}".format (args.data))
    print ("data size : {}".format (len (answers)))
    print ("average num of words for each sample : {}".format (q_len_mean))

    return questions, answers, ref

class MyDataset (Dataset):
    def __init__(self, args):
        super ().__init__ ()
        self.questions, self.answers, self.ref = data_reader (args)
        self.len = len (self.questions)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        reference = self.ref[index]
        return input, output, reference
    
def setup_data_loader(args):
    fix_seed (args.seed)
    worker_seed = torch.initial_seed () % 2 ** 32
    print ("worker_seed : {}".format (worker_seed))

    def seed_worker(worker_id):
        np.random.seed (worker_seed)
        random.seed (worker_seed)

    g = torch.Generator ()
    g.manual_seed (worker_seed)

    dataloader_num_workers = multiprocessing.cpu_count ()
    dataloader_num_workers = min (dataloader_num_workers, args.max_num_worker)
    print ("dataloader_num_workers: " + str (dataloader_num_workers))

    dataset = MyDataset (args)

    dataloader = torch.utils.data.DataLoader (dataset,
                                              shuffle=True,
                                              batch_size=args.batch_size,
                                              drop_last=False,
                                              num_workers=dataloader_num_workers,
                                              worker_init_fn=seed_worker,
                                              generator=g,
                                              pin_memory=True
                                              )

    return dataloader

def is_string_convertible(s):
    try:
        float (s)
        return True
    except:
        return False
    
def clean(predict):
    if len(predict)==0:
        return -1
    predicts = re.split (r"[Tt]he correct answer is", predict)
    answer_flag = True if len (predicts) > 1 else False

    if answer_flag:
        match = re.search(r'[AB]', predicts[-1])
        if predicts[-1][:6] == "A or B":
            return -1 
        if predicts[-1][:6] == "B or A":
            return -1
        if match:
            answer = match.group() 
            if answer == 'A':
                return 0
            else:
                return 1
        else:
            return -1
    else:
        return -1

def clean_judges(judgements,right_answer):
    reward = []
    predict_answer = [clean(predict) for predict in judgements]
    for i in range(len(judgements)):
        if predict_answer[i] == right_answer[i]:
            reward.append(float(1))
        else:
            reward.append(float(-1))
    
    return reward

def answer_cleaning(args, predict, must_choice=False):
    #predicts = predict.split ("\n")
    #predict = predicts[-1]

    predicts = re.split (r"[Tt]he answer is ", predict)
    answer_flag = True if len (predicts) > 1 else False
    #print(predicts)
    predict = predicts[-1]

    predict = predict.replace (",", "")
    predict = [s for s in re.findall (r'-?\d+\.?\d*', predict)]

    if len (predict) == 0:
        predict = ""
    else:
        if answer_flag:
            predict = predict[0]
        else:
            predict = predict[-1]

    if len(predict) == 0:
        predict = "No answer!"
    elif predict[-1] == ".":
        predict = predict[:-1]

    return predict


def new_history(history, viewpoint, flag):
    for i in range(len(history)):
        if history[i] == "None":
            if flag == 1:
                history[i] = "Student's Response: " + viewpoint[i]
            elif flag == 0:
                history[i] = "Problem solver's Response: " + viewpoint[i]
        elif flag == 1:
            history[i] += "\Student's Response: " + viewpoint[i]
        elif flag == 0:
            history[i] += "\Problem solver's Response: " + viewpoint[i]

    return history

def clean2(predict):
    if len(predict)==0:
        return -1
    predicts = re.split (r"[Ii] agree that the ", predict)
    answer_flag = True if len (predicts) > 1 else False

    if answer_flag:
        match = re.search(r'opponent is right.|proponent is right.', predicts[-1], re.IGNORECASE)
        if match:
            
            answer = match.group().lower()  # 将匹配结果转换为小写，以便统一处理
            if answer == 'opponent is right.':
                return 0
            
            elif answer == 'proponent is right.':
                return 1
            
            else:
                return -1
        else:
            return -1
    else:
        return -1

def clean_judges2(judgements):
    reward = []
    predict_answer = [clean2(predict) for predict in judgements]

    for i in range(len(judgements)):
        if predict_answer[i] == -1:
            reward.append(float(0))
        elif predict_answer[i] == 1:
            reward.append(float(1))
        else:
            reward.append(float(-1))
    
    return reward

def get_position(pre, gt_answer):
    if is_string_convertible(pre) and is_string_convertible(gt_answer):
        if float(pre) == float(gt_answer):
            return 1
        else:
            return 0
    return 0

def numpy_concatenate(np1, np2, position):
    re = np.where(position == 0, np1, np2)
    return re

# 判断gpt4的单个response中是否包含opponent win/proponent win
def clean3(predict):
    if len(predict)==0:
        return -2
    predicts = re.split (r"[Ii] declare: ", predict)
    answer_flag = True if len (predicts) > 1 else False

    if answer_flag:
        match = re.search(r'opponent win|proponent win|debate continue', predicts[-1], re.IGNORECASE)
        if match:
            answer = match.group().lower()  # 将匹配结果转换为小写，以便统一处理
            if answer == 'opponent win':
                return -1
            elif answer == 'proponent win':
                return 1
            elif answer == 'debate continue':
                return 0
        else:
            return -2
    else:
        return -2
    
def clean_judges3(judgements):
    reward = []
    predict_answer = [clean3(predict) for predict in judgements]
    for i in range(len(judgements)):
        if predict_answer[i] == 1:
            reward.append(float(1))
        elif predict_answer[i] == -1:
            reward.append(float(-1))
        else:
            reward.append(float(0))
    return reward

def split_reward(final_reward,terminate_pos,gamma):
    reward_dense = []
    for i in range(len(final_reward)):
        for k in range(terminate_pos[i]+1):
            T = terminate_pos[i]
            t = k
            reward = final_reward[i]*(1 - gamma) * (gamma**(T - t)) / (1 - gamma**(T + 1))
            reward_dense.append(torch.tensor(reward))

    return reward_dense

def split_reward2(final_reward,terminate_pos,gamma):
    reward_dense = []
    for i in range(len(final_reward)):
        if final_reward[i]==0:
            continue
        for k in range(terminate_pos[i]+1):
            T = terminate_pos[i]
            t = k
            reward = final_reward[i]*(1 - gamma) * (gamma**(T - t)) / (1 - gamma**(T + 1))
            reward_dense.append(torch.tensor(reward))

    return reward_dense


def clean_gpt_decision(predict):
    if len(predict)==0:
        return 0
    predicts = re.split (r"[Tt]he candidate answer is ", predict)
    answer_flag = True if len (predicts) > 1 else False

    if answer_flag:
        match = re.search(r'correct|wrong', predicts[-1], re.IGNORECASE)
        if match:
            answer = match.group().lower()  # 将匹配结果转换为小写，以便统一处理
            if answer == 'wrong':
                return 0
            elif answer == 'correct':
                return 1
        else:
            return 0
    else:
        return 0
    

def clean_model_decision(predict):
    if len(predict)==0:
        return -1
    predicts = re.split (r"[Tt]he candidate answer is", predict)
    answer_flag = True if len (predicts) > 1 else False

    if answer_flag:
        match = re.search(r'correct|incorrect', predicts[-1], re.IGNORECASE)
        if match:
            answer = match.group().lower()  # 将匹配结果转换为小写，以便统一处理
            if answer == 'incorrect':
                return 0
            elif answer == 'correct':
                return 1
        else:
            return -1
    else:
        return -1
    


def data_reader_testreward(args):
    questions = []
    answers = []
    pos = []
    ref = []
    decoder = json.JSONDecoder ()


    path = args.data
    with open (path) as f:
        lines = f.readlines ()
        for line in lines:
            json_res = decoder.raw_decode (line)[0]
            questions.append (json_res["question"].strip ())
            answers.append (json_res["candidate"].strip ())
            pos.append (json_res["flag"])
            if "/datasets/gsm8k/test/" in path:
                ref.append("nothing to reference.")
            else:
                ref.append (json_res["ref"])

    q_len_list = []
    for q in questions:
        q_len_list.append (len (q.split (" ")))
    q_len_mean = mean (q_len_list)
    
    print ("dataset : {}".format (args.data))
    print ("data size : {}".format (len (answers)))
    print ("average num of words for each sample : {}".format (q_len_mean))

    return questions, answers, pos, ref

class MyDataset2 (Dataset):
    def __init__(self, args):
        super ().__init__ ()
        self.questions, self.answers, self.pos, self.ref = data_reader_testreward (args)
        self.len = len (self.questions)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        input = self.questions[index]
        candidate = self.answers[index]
        pos = self.pos[index]
        ref = self.ref[index]
        return input, candidate, pos, ref
    
def setup_data_loader2(args):
    fix_seed (args.seed)
    worker_seed = torch.initial_seed () % 2 ** 32
    print ("worker_seed : {}".format (worker_seed))

    def seed_worker(worker_id):
        np.random.seed (worker_seed)
        random.seed (worker_seed)

    g = torch.Generator ()
    g.manual_seed (worker_seed)

    dataloader_num_workers = multiprocessing.cpu_count ()
    dataloader_num_workers = min (dataloader_num_workers, args.max_num_worker)
    print ("dataloader_num_workers: " + str (dataloader_num_workers))

    dataset = MyDataset2 (args)

    dataloader = torch.utils.data.DataLoader (dataset,
                                              shuffle=True,
                                              batch_size=args.batch_size,
                                              drop_last=False,
                                              num_workers=dataloader_num_workers,
                                              worker_init_fn=seed_worker,
                                              generator=g,
                                              pin_memory=True
                                              )

    return dataloader


def text_clean(list_of_str):
    re = []
    for text in list_of_str:
        tmp = text.strip()
        if tmp[-4:] == "</s>":
            tmp = tmp[:-4]
        if tmp[:18] == "Student's response:":
            tmp = tmp[18:]
        if tmp[:18] == "student's response:":
            tmp = tmp[11:]
        if tmp[:26] == "Problem solver's response:":
            tmp = tmp[26:]
        if tmp[:26] == "problem solver's response:":
            tmp = tmp[26:]    

        re.append(tmp.strip())

    return re


def clean_judges4(truths,lies):
    reward = []
    for i in range(len(truths)):
        truth = truths[i]
        lie = lies[i]
        
        if (len(truth)==0) | (len(lie)==0):
            reward.append(float(0))
            continue

        truth_predicts = re.split (r"[Ii] agree that the", truth)
        lie_predicts = re.split (r"[Ii] agree that the", lie)
        if len (truth_predicts) > 1:
            match = re.search(r'student is right.', truth_predicts[-1], re.IGNORECASE)
            if match:
                answer = match.group().lower()
                if answer == 'student is right.':
                    reward.append(float(-1))
                    continue
                
        if len (lie_predicts) > 1:
            match = re.search(r'problem solver is right', lie_predicts[-1], re.IGNORECASE)
            if match:
                answer = match.group().lower()
                if answer == 'problem solver is right':
                    reward.append(float(1))
                    continue
                else:
                    reward.append(float(0))
                    continue
            else:
                reward.append(float(0))
                continue
        else:
            reward.append(float(0))
        
    return reward

def judge_reward(judge, pos):
    if len(judge) != 0:
        pre = re.split (r"[Tt]he candidate answer is", judge)
        if len (pre) > 1:
            match = re.search(r'correct|incorrect', pre[-1], re.IGNORECASE)
            if match:
                answer = match.group().lower()  # 将匹配结果转换为小写，以便统一处理
                if (answer == 'incorrect') & (pos==0):
                    return 1.0
                if (answer == 'correct') & (pos==1):
                    return 1.0
                else:
                    return -1.0
            else:
                return -1.0
        else:
            return -1.0
    else:
        return -1.0



def data_reader_rewrite(args):
    questions = []
    answers = []
    ref = []
    decoder = json.JSONDecoder ()


    path = args.data
    with open (path) as f:
        lines = f.readlines ()
        for line in lines:
            json_res = decoder.raw_decode (line)[0]
            questions.append (json_res["question"].strip ())
            answers.append (json_res["candidate"].strip ())
            ref.append (json_res["re_ref"].strip ())

    q_len_list = []
    for q in questions:
        q_len_list.append (len (q.split (" ")))
    q_len_mean = mean (q_len_list)
    
    print ("dataset : {}".format (args.data))
    print ("data size : {}".format (len (answers)))
    print ("average num of words for each sample : {}".format (q_len_mean))

    return questions, answers, ref

class MyDataset3 (Dataset):
    def __init__(self, args):
        super ().__init__ ()
        self.questions, self.answers, self.ref = data_reader_rewrite (args)
        self.len = len (self.questions)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        input = self.questions[index]
        candidate = self.answers[index]
        re_ref = self.ref[index]
        return input, candidate, re_ref
    
def setup_data_loader3(args):
    fix_seed (args.seed)
    worker_seed = torch.initial_seed () % 2 ** 32
    print ("worker_seed : {}".format (worker_seed))

    def seed_worker(worker_id):
        np.random.seed (worker_seed)
        random.seed (worker_seed)

    g = torch.Generator ()
    g.manual_seed (worker_seed)

    dataloader_num_workers = multiprocessing.cpu_count ()
    dataloader_num_workers = min (dataloader_num_workers, args.max_num_worker)
    print ("dataloader_num_workers: " + str (dataloader_num_workers))

    dataset = MyDataset3 (args)

    dataloader = torch.utils.data.DataLoader (dataset,
                                              shuffle=True,
                                              batch_size=args.batch_size,
                                              drop_last=False,
                                              num_workers=dataloader_num_workers,
                                              worker_init_fn=seed_worker,
                                              generator=g,
                                              pin_memory=True
                                              )

    return dataloader