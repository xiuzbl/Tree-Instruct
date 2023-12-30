import os, sys
import json
# from gptutils import ChatGPTWrapper
import logging
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
import tqdm
import ast
import time
import openai
# from process_generated_tree import oridata, all_instructions
# from generated_tree import visited


openai.api_key = 'your_openai_api_key'


def have_processed(old_instruction):
    old_idx = all_instructions.index(old_instruction)
    if old_idx in visited:
        return True
    else:
        return False

from langdetect import detect

def is_english(sentence):
    try:
        lang = detect(sentence)
        return lang == 'en'
    except:
        return False

def worker(sample):
    prompt0 = """You are an instruction rewriter. You need to rewrite a given user instruction following Procedure and all the Requirements. You MUST ONLY return the NEW instruction you rewrite.

Procedure:
step-1: Parse the old "instruction" to a TREE-1 through semantic parsing in natural language processing.
step-2: EXPAND the above NEW TREE-1 from DEPTH or WIDTH by ADDING Three meaningful NEW NODEs as Nouns or Verbs to form a NEW TREE-2. The new nodes should be constructed with detailed and pertinent information.
step-3: Generate a totally NEW "instruction" based on the expanded NEW TREE-2. 

Requirements:
1. Generate ONLY ONE NEW "INSTRUCTION" based on old "instruction" following "Procedure" in at most 100 words. 
2. Make sure the new "instruction" is reasonable, clear, more COMPLEX and at least 12 words longer than the old "instruction". 
2.1 Do not generate new nodes more than required.
3. Maintain the DATA FORMAT and the LANGUAGE of the old instruction. 
4. NOT return any sentences or steps in the "Procedure" or "Requirements": e.g., "adding three", "tree-1,2", "semantic parsing" are NOT allowed.
5. NO details or explanations for the instruction generation process.
6. NO response to the old or new instructions.
7. NO URLs or websites appears unless the old instruction has them.

Old instruction: \"{{sample}}\"

New instruction: """

    if sample['id'] in visited:
        return sample, 'HAVE-VISITED'


    instruction = sample['new']

    if '\"' == instruction[0]:
        instruction = instruction[1:]
    if '\"' == instruction[-1]:
        instruction = instruction[:-1]
    prompt = prompt0.replace('{{sample}}', instruction.lstrip().rstrip())

    messages = [{'role': 'user', 'content': prompt}]

    try:
        for i in range(10):
            out = openai.ChatCompletion.create(model="gpt-4",
                                               messages=messages,
                                               max_tokens=400)
            if out is None:
                raise Exception('Call output is NONE. Need retry...')
            else:
                response = out['choices'][0]['message']['content']
                if len(response) > 0: break

    except Exception as e:
        # print(f'----------------{e}===================')
        try:
            for i in range(50):
                out = openai.ChatCompletion.create(model="gpt-4",
                                                messages=messages,
                                                max_tokens=400)
                if out is not None and len(out['choices'][0]['message']['content']):
                    response = out['choices'][0]['message']['content']
                    if len(response) > 0:
                        break
            if out is None:
                response = "NONE"
        except:
            response = 'NONE'


    logging.info(f"Response::{response}")
    return sample, response

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    num = 2000

    dirname = 'alpaca'

    date = 'tree_6nodes'

    beginidx = 0

    outname = 'tree1k_6nodes_instruction'
    os.makedirs(os.path.join("./dataout/", dirname, date), exist_ok=True)


    inputfile = 'alpaca_1000.json'
    inputfile = 'tree1k_3nodes_instruction.json'

    with open(inputfile,'r') as f:
        inputs = [json.loads(s) for s in f.readlines()] 

    while True:
        output_path = os.path.join("./dataout/", dirname, date, outname+'.json')
        fp = open(output_path, 'a+')
    
        outdata = []
        with open(output_path, 'r') as f:
            for s in f.readlines():
                print(s)
                outdata.append(json.loads(s))

        visited = set()
        for i in range(len(outdata)):
            if outdata[i]['new'] != 'NONE' and 'semantic parsing' not in outdata[i]['new']:
                new = outdata[i]['new']
                old = outdata[i]['old']
                len_new = len(outdata[i]['new'].split(' '))
                len_old = len(outdata[i]['old'].split(' '))

                visited.add(outdata[i]['id'])
        print(f'Have processed {len(visited)}')

        inputs = [s for s in inputs[beginidx:num] if s['id'] not in visited]
        if 0 == len(inputs):
            print('Have processed all!')
            break 

        print(type(inputs))
        print(f'Number of input files: {len(inputs)}')

        with ThreadPool(10) as pool:
            for sample, new_sample in pool.imap(worker, inputs):
                if new_sample != 'HAVE-VISITED' and new_sample!='NONE' and 'undefinedsome thing error, try again later' not in new_sample:
                    res = {}
                    res['new'] = new_sample
                    res['old'] = sample['new'] # from alpaca-tree-3-nodes
                    res['input'] = sample['input']
                    res['output'] = sample['output']
                    res['id'] = sample['id']
                    res['num_turns'] = 1
                    res['model'] = 'gpt4'
                    print(json.dumps(res, ensure_ascii=False), file=fp, flush=True)
                    fp.flush()
        fp.close()
        print(f'Congrats~')
