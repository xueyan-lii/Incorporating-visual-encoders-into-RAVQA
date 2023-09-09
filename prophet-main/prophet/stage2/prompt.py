# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Runner that handles the prompting process
# ------------------------------------------------------------------------------ #

import os, sys
# sys.path.append(os.getcwd())
import csv
import pickle
import json, time
import math
import random
import argparse
from datetime import datetime
from copy import deepcopy
import yaml
from pathlib import Path
import openai

from .utils.fancy_pbar import progress, info_column
from .utils.data_utils import Qid2Data
from configs.task_cfgs import Cfgs


class Runner:
    def __init__(self, __C, evaluater):
        self.__C = __C
        self.evaluater = evaluater
        openai.api_key = __C.OPENAI_KEY
    
    def gpt3_infer(self, prompt_text, _retry=0):
        # print(prompt_text)
        # exponential backoff
        if _retry > 0:
            print('retrying...')
            st = 2 ** _retry
            time.sleep(st)
        
        if self.__C.DEBUG:
            # print(prompt_text)
            #time.sleep(0.0005)
            return '0', 0

        try:
            # print('calling gpt3...')
            
            response = openai.ChatCompletion.create(
                model=self.__C.MODEL,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=self.__C.TEMPERATURE,
                max_tokens=self.__C.MAX_TOKENS,
                top_p=1,
                stop=["\n", "<|endoftext|>"],
                # timeout=20,
            )
            '''
            response = openai.Completion.create(
                engine=self.__C.MODEL,
                prompt=prompt_text,
                temperature=self.__C.TEMPERATURE,
                max_tokens=self.__C.MAX_TOKENS,
                logprobs=1,
                #n=1,
                #stream=False,
                #top_p=1,
                stop=["\n", "<|endoftext|>"],
                # timeout=20,
            )
            '''
        except Exception as e:
            print(type(e), e)
            if str(e) == 'You exceeded your current quota, please check your plan and billing details.':
                exit(1)
            return self.gpt3_infer(prompt_text, _retry + 1)

        response_txt = response.choices[0].message.content.strip()
        #response_txt = response.choices[0].text.strip()
        # print(response_txt)
        prob=0
        
        return response_txt, prob
    
    def run(self):
        ## where logs will be saved
        Path(self.__C.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(self.__C.LOG_PATH, 'w') as f:
            f.write(str(self.__C) + '\n')
        ## where results will be saved
        Path(self.__C.RESULT_DIR).mkdir(parents=True, exist_ok=True)
        
        self.cache = {}
        self.cache_file_path = os.path.join(
            self.__C.RESULT_DIR,
            'cache.json'
        )
        if self.__C.RESUME:
            self.cache = json.load(open(self.cache_file_path, 'r'))
        
        print('Note that the accuracies printed before final evaluation (the last printed one) are rough, just for checking if the process is normal!!!\n')
        '''
        self.trainset = Qid2Data(
            self.__C, 
            self.__C.TRAIN_SPLITS,
            True
        )
        '''
        self.valset = Qid2Data(
            self.__C, 
            self.__C.EVAL_SPLITS,
            self.__C.EVAL_NOW,
            self.__C.EXAMPLES_PATH,
        )

        infer_times = self.__C.T_INFER #1
        
        for qid in progress.track(self.valset.qid_to_data, description="Working...  "):
            if qid in self.cache:
                answer = self.cache[qid]['answer']
                self.evaluater.add(qid, answer)
                continue
            ques = self.valset.get_question(qid)
            
            prompt_text = self.valset.get_similar_qids(qid)

            prompt_info_list = []
            ans_pool = {}
            # multi-times infer
            for t in range(infer_times): #1
                # print(f'Infer {t}...')
                gen_text, gen_prob = self.gpt3_infer(prompt_text)
                #print('answer is',gen_text)

                ans = self.evaluater.prep_ans(gen_text)
                if ans != '':
                    ans_pool[ans] = ans_pool.get(ans, 0.) + gen_prob

                prompt_info = {
                    'prompt': prompt_text,
                    'answer': gen_text,
                    'confidence': gen_prob
                }
                prompt_info_list.append(prompt_info)
                #time.sleep(0.5)
            
            # vote
            ''''
            if len(ans_pool) == 0:
                answer = self.valset.get_topk_candidates(qid, 1)[0]['answer']
            else:
                answer = sorted(ans_pool.items(), key=lambda x: x[1], reverse=True)[0][0]
            '''
            answer = gen_text
            self.evaluater.add(qid, answer)
            self.cache[qid] = {
                'question_id': qid,
                'answer': answer,
                'prompt_info': prompt_info_list
            }
            json.dump(self.cache, open(self.cache_file_path, 'w'))
            '''
            ll = len(self.cache)
            if self.__C.EVAL_NOW:
                if ll > 21 and ll % 10 == 0:
                    rt_accuracy = self.valset.rt_evaluate(self.cache.values())
                    info_column.info = f'Acc: {rt_accuracy}'
            '''
        '''
        #check that scores is the same, which is 
        with open("/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/prophet-main/outputs/24574008.csv") as csv_file_handler:
            csv_reader = csv.DictReader(csv_file_handler)
            for rows in csv_reader:
                #qid_to_data[rows['question_id']]['similar_qids'] = rows['prompt']
                self.evaluater.add(rows['question_id'], rows['prediction'])
        '''    
        self.evaluater.save(self.__C.RESULT_PATH)
        if self.__C.EVAL_NOW:
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                self.evaluater.evaluate(logfile)
        
def prompt_login_args(parser):
    parser.add_argument('--debug', dest='DEBUG', help='debug mode', action='store_true')
    parser.add_argument('--resume', dest='RESUME', help='resume previous run', action='store_true')
    parser.add_argument('--task', dest='TASK', help='task name, e.g., ok, aok_val, aok_test', type=str, required=True)
    parser.add_argument('--version', dest='VERSION', help='version name', type=str, required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', type=str, default='configs/prompt.yml')
    parser.add_argument('--examples_path', dest='EXAMPLES_PATH', help='answer-aware example file path, default: "assets/answer_aware_examples_for_ok.json"', type=str, default=None)
    parser.add_argument('--openai_key', dest='OPENAI_KEY', help='openai api key', type=str, default=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heuristics-enhanced Prompting')
    prompt_login_args(parser)
    args = parser.parse_args()
    __C = Cfgs(args)
    with open(args.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    print(__C)

    runner = Runner(__C)
    runner.run()
