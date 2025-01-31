import re
import base64
import requests
import json, os
import pandas as pd
import tqdm
from pathlib import Path
from pprint import pprint
import sys


# set parent directory to address relative imports
directory = Path(os.getcwd()).absolute()
sys.path.append(
    str(directory)
)  # note: no ".parent" addition is needed for python (.py) files
print(os.getcwd())

# import environment
from AI2Thor.env_new import AI2ThorEnv

# import utils for this baseline
from multi_react_utils import *
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

with open(os.path.expanduser("~") + "/openai_key.json") as json_file:
    key = json.load(json_file)
    api_key = key["my_openai_api_key"]
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

# autoconfig
auto = AutoConfig(config_file=args.config_file)
# auto.set_task(args.task)
# auto.set_floorplan(args.floorplan)
# timeout = auto.get_task_timeout()

# to avoid warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# no warnings
warnings.filterwarnings("ignore")

# environment initialization
# config = auto.config()

# amt_tasks=auto.get_amt_tasks()
amt_tasks = [3] # "1_put_computer_book_remotecontrol_sofa"
for i in range(len(amt_tasks)):
    task_index = amt_tasks[i]

    auto.set_task(task_index)
    amt_floorplans=auto.get_amt_floorplans(task_index)

    for fp_index in range(amt_floorplans):

        auto.set_floorplan(fp_index)
        timeout=auto.get_task_timeout()
        config = auto.config()

        max_retries = 3
        retries = 0

        while retries < max_retries:
            try: # sometimes rare errors happen, so give each task and floorplan a few tries

                env = AI2ThorEnv(auto.config())
                task = auto.task_string()
                d = env.reset(task=task)

                user_prompt = f"Task: {task}"
                summariserLLM = ObsSummaryLLM(user_prompt, headers)

                logger=Logger('New_ReAct', env) 
                for step_num in range(1, timeout+1):
                    response, user_prompt = get_gpt_response(env, config, user_prompt, step_num, summariserLLM )
                    outdict = get_action(response)
                    # print('outdict', outdict)
                    # get closest feasible action
                    action_texts = [outdict[agent_name[0]], outdict[agent_name[1]]]
                    action = action_checker(env, action_texts)
                    # execute action in environment
                    d, action_successes = env.step(action)
                    # update user prompt with action taken and its success
                    user_prompt = prepare_prompt_post_action(env,user_prompt, action_texts, action, action_successes)
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
                        finished=finished)
                    print('_'*50)
                    print(f"Step {step_num}")
                    print(f"Completed Subtasks: ")
                    print("\n".join(env.checker.subtasks_completed))
                    # if the model outputs "Done" for both agents, break
                    if all(status == 'Done' for status in action):
                        break

                env.controller.stop()
                break
            except Exception as e:
                print(f"Error occurred: {e}. Retrying...")
                env.controller.stop()
                retries += 1
                if retries >= max_retries:
                    print("Max retries reached. Moving to next floorplan or task.")
                    break


