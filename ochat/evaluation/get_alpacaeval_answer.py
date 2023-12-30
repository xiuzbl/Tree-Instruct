import os
import json
import argparse
import time
import asyncio
import datasets
from ochat.config.model_config import MODEL_CONFIG_MAP
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import Optional
from glob import glob
from copy import deepcopy
import orjson
import openai
from tqdm import tqdm
from openai.error import RateLimitError, ServiceUnavailableError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from vllm import LLM, SamplingParams

from ochat.config.model_config import MODEL_CONFIG_MAP
from ochat.evaluation.match_answer import MATCH_ANSWER_FUNCTION
from ochat.evaluation.conversation_templates import CONVERSATION_TEMPLATES

openai.api_key = "EMPTY"
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
async def chat_completion_with_backoff(sem, **kwargs):
    async with sem:
        return await openai.ChatCompletion.acreate(**kwargs)

async def get_model_answers(
    questions: list,
    # questions: dict,
    model_type: str,
    model_path: str,
    exp: str,
    max_out_token: int
):
    # Init vLLM engine
    model_config = MODEL_CONFIG_MAP[model_type]

    engine = LLM(model_path,
                 max_num_batched_tokens=model_config.model_max_context,
               )
    sampling_params = SamplingParams(temperature=0.7,
                                    #  max_tokens=model_config.model_max_context,
                                    max_tokens=max_out_token,
                                     stop=[model_config.eot_token])

    # Init tokenizer
    tokenizer = model_config.model_tokenizer_create(model_path)

    def _tokenize(text):
        """Tokenize text-only, ignoring all special tokens."""
        return tokenizer.convert_tokens_to_ids(tokenizer._tokenize(text))

    def _tokenize_special(special_name):
        return tokenizer.convert_tokens_to_ids(special_name)

    # Complete
    prompts = []
    prompt_indices = []
    template_name = "openchat"
    # template_name = 'none'
    # for template_name, template_questions in questions.items():
    prompt_template_fn = CONVERSATION_TEMPLATES[template_name]
    for idx, q in enumerate(questions):

        # for idx, q in enumerate(template_questions):
        if "response" in q and q["response"]:
            continue

        tokens = prompt_template_fn(q["instruction"],
                                    model_config=model_config,
                                    tokenize_fn=_tokenize,
                                    tokenize_special_fn=_tokenize_special)

        # Truncate to specified tokens
        max_context = model_config.model_max_context
        if max_context is not None:
            tokens = tokens[-max_context:]

        prompt_indices.append((template_name, idx))
        prompts.append(tokens)

        q["prompt"] = tokenizer.decode(tokens)

    # calculate & fill in responses
    responses = engine.generate(prompt_token_ids=prompts, sampling_params=sampling_params)

    responses = sorted(responses, key=lambda x: int(x.request_id))
    responses = [x.outputs[0].text for x in responses]

    res = []
    for (template_name, idx), resp in zip(prompt_indices, responses):
        # questions[template_name][idx]["response"] = resp
        # questions[template_name][idx]
        ss = {}
        ss['instruction'] = questions[idx]['instruction']
        ss['output'] = resp
        ss['dataset'] = questions[idx]['dataset']
        ss['template_name'] = template_name
        ss['generator'] = exp
        ss['id'] = idx
        res.append(ss)

    return res

async def run_eval(
    max_out_token: int,
    model_type: str,
    model_path: str,
    conversation_templates: list,
    max_num: int, 
    exp: str, 
    # data_path: str,
    eval_sets: list,
    continue_from: Optional[str],
    output_file: str,
    parallel: int
):
    data = datasets.load_dataset(path='../TreeData/evaldata/alpaca_eval', name="alpaca_eval")["eval"]
    questions = []
    print(data[0], flush=True)
    for i in range(max_num):
        sample = data[i]
        # print(f'sample {sample}', flush=True)
        questions.append(sample)

    questions = await get_model_answers(questions, model_type, model_path, exp, max_out_token)
    # Write results
    # if output_file is None:
    output_file = os.path.join(output_file,  f"eval_{exp}.json")

    with open(output_file, "wb") as f:
        f.write(orjson.dumps(questions, option=orjson.OPT_INDENT_2))


async def main():
    parser = argparse.ArgumentParser()

    # Input / output
    parser.add_argument('--max_out_token', type=int)
    parser.add_argument('--max_num', type=int)
    parser.add_argument('--exp', type=str)
    parser.add_argument("--model_type",             type=str, default="gpt-3.5-turbo")
    parser.add_argument("--model_path",             type=str, default=None)
    parser.add_argument("--conversation_templates", type=str, nargs="+", default=["default"])

    # parser.add_argument("--data_path", type=str, default="ochat/evaluation/eval_data")
    parser.add_argument("--eval_sets", type=str, nargs="+", default=[])

    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--output_file",   type=str, default=None)
    parser.add_argument("--parallel",      type=int, default=16)

    args = parser.parse_args()

    await run_eval(**vars(args))


if __name__ == "__main__":
    asyncio.run(main())
