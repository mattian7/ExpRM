# 对rewrite后的数据进行预处理，删除掉思维链中的分析部分，只保留最后合成的正确答案
import json
import re
def transform_jsonl(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            record = json.loads(line)
            new_record = {}
            pres = re.split (r'\[Fake Answer\]:', record["re-ref"])
            if len(pres) > 1:
                fake = pres[-1].strip()
                pres_0 = re.split (r"Let's think step by step.", fake)
                if len(pres_0) > 1:
                    new_record["re_ref"] = pres_0[-1].strip()
                new_record["candidate"] = record["candidate_answer"][25:]
                new_record["question"] = record["question"]
            else:
                continue  # 如果flag字段不是'tensor(0)'或'tensor(1)'，跳过该记录

            outfile.write(json.dumps(new_record) + "\n")

# 输入文件路径
input_file = './datasets/gsm8k/train/rewritten_all.jsonl'
# 输出文件路径
output_file = './datasets/gsm8k/train/rewritten_final.jsonl'

# 调用函数进行转换
transform_jsonl(input_file, output_file)