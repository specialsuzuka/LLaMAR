import re
import base64
import requests
import json, os
import pandas as pd
import tqdm
from pathlib import Path
import sys
import ast
from pprint import pprint
import time
import math

# set parent directory to address relative imports
directory = Path(os.getcwd()).absolute()
sys.path.append(
    str(directory)
)  # note: no ".parent" addition is needed for python (.py) files
pprint(os.getcwd())

# import environment
from AI2Thor.env_new import AI2ThorEnv

# import utils for this baseline
from act_utils import *
from AI2Thor.baselines.utils import Logger, AutoConfig
from AI2Thor.summarisers.obs_summariser_2 import ObsSummaryLLM

import warnings
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=int, default=0)
parser.add_argument("--floorplan", type=int, default=0)
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--config_file", type=str, default="config.json")
args = parser.parse_args()

# to avoid warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# no warnings
warnings.filterwarnings("ignore")


with open(os.path.expanduser("~") + "/openai_key.json") as json_file:
    key = json.load(json_file)
    api_key = key["my_openai_api_key"]
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

# autoconfig
auto = AutoConfig(config_file=args.config_file)
auto.set_task(args.task)
auto.set_floorplan(args.floorplan)
timeout = auto.get_task_timeout()

# environment initialization
config = auto.config()
env = AI2ThorEnv(config)
task = auto.task_string()
d = env.reset(task=task)
user_prompt = f"Task: {task}"
summariserLLM = ObsSummaryLLM(user_prompt, headers)

# logger
logger = Logger(env=env, baseline_name="act")
pprint(f'baseline path (w/ results): {logger.baseline_path}')

pprint("*" * 50)
pprint("Starting the act baseline")
pprint("*" * 50)

MAX_RETRIES = 10
RATE_LIMIT_WAIT_DEFAULT = 5

failed=False
for step_num in tqdm.trange(timeout):
    retries = 0
    rate_limit_retries = 0
    while retries < MAX_RETRIES:
        try: # sometimes rare errors happen, so give it a few tries
            response, user_prompt = get_gpt_response(env, config, user_prompt, step_num, summariserLLM)
            if response.status_code == 429:  # HTTP status code for rate limiting
                pprint(f"429 {response.json()}")
                min_wait = 1 + math.ceil(float(response.headers.get('Retry-After', RATE_LIMIT_WAIT_DEFAULT)))
                mult_factor = 2 ** rate_limit_retries
                retry_after = min_wait * mult_factor
                pprint(f"Rate limit reached. Waiting for {retry_after} seconds...")
                pprint("start")
                time.sleep(retry_after)
                pprint("end")
                rate_limit_retries += 1
                continue
            outdict = get_action(response)
            break
        except Exception as e:
            if 'rate_limit_exceeded' in str(e):
                pprint(f"Rate limit exceeded: {e}. Retrying...")
                rate_limit_retries += 1
            else:
                pprint(f"Error occurred: {e}. Retrying...")
                if response:
                    pprint(f"response JSON {response.json()}")
                if outdict:
                    pprint(f"outdict {outdict}")
                retries += 1
                if retries >= MAX_RETRIES:
                    pprint("Task failed, max retries reached.")
                    failed=True
                    break
    if failed:
        break
    # get closest feasible action
    action_texts = [outdict[agent_name[0]], outdict[agent_name[1]]]
    action = action_checker(env, action_texts)
    # execute action in environment
    d, action_successes = env.step(action)
    # update user prompt with action taken and its success
    user_prompt = prepare_prompt_post_action(env, user_prompt, action_texts, action, action_successes)
    # append to dataframe
    coverage = env.checker.get_coverage()
    transport_rate = env.checker.get_transport_rate()
    finished = env.checker.check_success()
    # logging
    logger.log_step(
        step=step_num,
        preaction=action_texts,
        action=action,
        success=action_successes,
        coverage=coverage,
        transport_rate=transport_rate,
        finished=finished
        )
    if args.verbose:
        pprint('_'*50)
        pprint(f"Step {step_num}")
        pprint(f"Completed Subtasks: ")
        pprint("\n".join(env.checker.subtasks_completed))
    # if the model outputs "Done" for both agents, break
    if all(status == 'Done' for status in action):
        break

env.controller.stop()
