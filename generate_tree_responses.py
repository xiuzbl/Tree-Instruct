import os, sys
import json
# from gptutils import ChatGPTWrapper
import logging
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
import tqdm
import ast
import openai
# from process_generated_response import all_instructions
# from response import visited


openai.api_key = 'your_openai_api_key'

def worker(sample):
    prompt0 = """You are an Assistant, you need to generate a response for a given instruction by a User:

You can refer to the following example:
</example>
example-Instruction: Explain why 4/16 is equivalent to 1/4.
example-Response: The fraction 4/16 is equivalent to 1/4 because both fractions represent the same value. A fraction can be simplified by dividing both the numerator and the denominator by a common factor. In this case, 4 is a common factor of both the numerator and the denominator of 4/16. When we divide both by 4, we get 4/4 = 1 and 16/4 = 4, so the simplified fraction is 1/4. Alternatively, we can think of this in terms of multiplication. For example, if we multiply the numerator and denominator of the fraction 1/4 by 4, we get (1x4)/(4x4), or 4/16. Since both fractions can be derived from the other through multiplication or division by the same number, they represent the same value and are equivalent.
</example>

You MUST:
- Carefully understand the following 'Instruction' that describes a task. 
- Provide a pricise response NO MORE THAN 1000 words to complete the task described in the 'Instruction'. 
- If the task involves complexity and requires reasoning, you can break it down to sub-tasks and complete them step-by-step. 
- Make sure the response is concise, accurate, and FULLY meets the requirements outlined in the 'Instruction'.
- If your response is short, also give the explanation.
- NO URL or websites in the response, unless the instruction asks for it.

Instruction: \"{{instruction}}\"

Response: """

    prompt = prompt0.replace('{{instruction}}', sample['instruction'])
    messages = [
        {'role': 'user', 'content': prompt}
    ]
    logging.info(f"Calling ChatGPT")
    try:
        for i in range(50):
            out = openai.ChatCompletion.create(model="gpt-4",
                                               messages=messages,
                                               max_tokens=1500)
            if out is None:
                raise Exception('Call output is NONE. Need retry...')
            else:
                response = out['choices'][0]['message']['content']
            if len(response) > 0:
                break
    except Exception as e:
        try:
            for i in range(50):
                out = openai.ChatCompletion.create(model="gpt-4",
                                                messages=messages,
                                                max_tokens=1500)
                if out is not None:
                    response = out['choices'][0]['message']['content']
                if len(response) > 0:
                    break
        except:
            response = 'NONE'

    logging.info(f"Response::{response}")
    return sample, response

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dirname = 'alpaca'

    outname = 'tree1k_6nodes_response'

    date = 'tree_6nodes'

    num = 1000
    begin_idx = 0

    inputfile = 'tree1k_6nodes_instruction.json'
    with open(inputfile,'r') as f:
        inputs = [json.loads(s) for s in f.readlines()]

    while True:
        output_path = os.path.join("./dataout/", dirname, date, outname + '.json')
        fp = open(output_path, 'a+')


        with open(output_path,'r') as f:
            responsedata = [json.loads(s) for s in f.readlines()]
    
        visited = set()

        for sample in responsedata:
            id = sample['id']
            response = sample['output']
            if id not in visited and response!='NONE' and 'undefinedplease retry later' not in response:
                visited.add(id)
    
        new_inputs = []
        for sample in inputs:
            id = sample['id']        
            if id not in visited:

                input = sample['input']
                new_instruction = sample['new']

                if '\"' == new_instruction[0]:
                    new_instruction = new_instruction[1:]
                if '\"' == new_instruction[-1]:
                    new_instruction = new_instruction[:-1]

                if input != '' and 'no input' not in input.lower():
                    new_instruction = new_instruction + f'\"{input}\"'
                ss = sample.copy()
                ss['instruction'] = new_instruction
                print(ss)

                new_inputs.append(ss)

        inputs = new_inputs[begin_idx:num]
        print(type(inputs))
        print(f'Number of input files: {len(inputs)}')

        if len(inputs)==0:
            print(f'Have already generated all responses')
            break
        else:
            with ThreadPool(10) as pool:

                for sample, response in pool.imap(worker, inputs):
                    if response != 'HAVE-VISITED' and response!= 'NONE' and 'undefinedplease retry' not in response:
                        res = {}
                        res['instruction'] = sample['instruction']
                        res['output'] = response
                        res['old_instruction'] = sample['old']
                        res['old_output'] = sample['output']
                        res['id'] = sample['id']
                        # res['replace_turn_id'] = sample['replace_turn_id']
                        res['num_turns'] = 1
                        res['model'] = sample['model']
                        res['items'] = []
                        res['items'].append({
                            # 'from': 'user',
                            'from': 'human',
                            'value': sample['instruction']
                        })
                        res['items'].append({
                            'from':'gpt',
                            'value': response
                        })
                        print(json.dumps(res, ensure_ascii=False), file=fp, flush=True)
                        fp.flush()
    
            fp.close()
    print(f'Congrats~')
