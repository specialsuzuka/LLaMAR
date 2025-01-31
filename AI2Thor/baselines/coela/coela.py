import json, os
import pandas as pd
import tqdm
from pathlib import Path
import sys

# set directory to top folder level to address relative imports
directory = Path(os.getcwd()).absolute()
sys.path.append(
    str(directory)
)  # note: no ".parent" addition is needed for python (.py) files

# import environment
from AI2Thor.env_new import AI2ThorEnv

# import utils for this baseline
from AI2Thor.baselines.utils import Logger, AutoConfig
from AI2Thor.baselines.coela.coela_utils import *

import warnings
import time
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=int, default=0)
parser.add_argument("--floorplan", type=int, default=0)
parser.add_argument("--verbose", type=bool, default=False)
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

a = time.time()

env = AI2ThorEnv(config)
d = env.reset(task=auto.task_string())
# logger
logger = Logger(env=env, baseline_name="coela")

b = time.time()

# some inits
previous_action = [] * config.num_agents
previous_success = [True, True]  # initialize with True
df = pd.DataFrame(columns=["Step", "Action", "Message", "Random Flag", "Pre Random Action"])

agent1 = CoELA_Agent(env, 0, "Alice", "Bob", auto.task_string(), 1000)
agent2 = CoELA_Agent(env, 1, "Bob", "Alice", auto.task_string(), 1000)
agents_list = [agent1, agent2]

print("_" * 50)
print("Starting CoELA baseline")
print("_" * 50)


def append_row(df, step, action, message, random_flag, pre_random_action):
    row = pd.DataFrame(
        [[step, action, message, random_flag, pre_random_action]],
        columns=["Step", "Action", "Message", "Random Flag", "Pre Random Action"],
    )
    df = pd.concat([df, row])
    return df


# there is weird try-except loop wrapper (done in order to prevent json errors)
for step_num in tqdm.trange(timeout):
    pre_actions = []
    actions = []
    flags = []
    pre_messages = []
    messages = []
    message_speaker = []
    for agent in agents_list:
        success = False
        while not success:
            try:
                (
                    agent_comm_sys_prompt,
                    agent_comm_user_prompt,
                ) = agent.prepare_comm_prompt(None)
                response = agent.get_gpt_response(
                    agent_comm_sys_prompt, agent_comm_user_prompt
                )
                success = True
            except:
                pass
        message = response.json()["choices"][0]["message"]["content"]
        pre_messages.append(message)
        # messages.append(message)

        success = False
        while not success:
            try:
                (
                    agent_planner_sys_prompt,
                    agent_planner_user_prompt,
                ) = agent.prepare_planner_prompt(message)
                response = agent.get_gpt_response(
                    agent_planner_sys_prompt, agent_planner_user_prompt
                )
                success = True
            except Exception as e:
                print(e)
                pass
        pre_action = response.json()["choices"][0]["message"]["content"]
        pre_actions.append(pre_action)
        action, is_message, flag = agent.process_action(pre_action)
        if is_message:
            action = f"SendMessage({action})"
            messages.append(action)
            message_speaker.append(agent.name)
        flags.append(flag)
        actions.append(action)

    # update the dialogue history for both agents
    for message, speaker in zip(messages, message_speaker):
        for agent in agents_list:
            agent.dialogues.append(message)
            agent.dialogues_speaker.append(speaker)
    print(actions, messages, flags, pre_actions)
    df = append_row(df, step_num, actions, pre_messages, flags, pre_actions)
    d1, successes = env.step(actions)
    # get statistics and finish step
    coverage = env.checker.get_coverage()
    transport_rate = env.checker.get_transport_rate()
    finished = env.checker.check_success()

    # log current 'step'
    logger.log_step(
        preaction = pre_actions,
        step=step_num,
        action=actions,
        success=successes,
        coverage=coverage,
        transport_rate=transport_rate,
        finished=finished,
    )

    if args.verbose:
        print("_" * 50)
        print(f"Step {step_num}")
        print(f"Completed Subtasks: ")
        print("\n".join(env.checker.subtasks_completed))

    # if the model outputs "Done" for both agents, break
    if all(status == "Done" for status in actions):
        break

act_message_path = logger.baseline_path / logger.env.action_dir_path
df.to_csv(act_message_path / str("coela_message.csv"), index=False)


# STOP - stop the controller / remove window
env.controller.stop()
