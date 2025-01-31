import re
import base64
import requests
import json, os
import pandas as pd
from pathlib import Path
import sys
import ast

# --- DESCRIPTION ---
# This is the utils file for the CoT baseline that defines the environment and functions used in CoT.py. This script is not directly executed!

# set parent directory to address relative imports
directory = Path(os.getcwd()).absolute()
sys.path.append(str(directory)) 

from AI2Thor.object_actions import get_closest_feasible_action, get_closest_object_id

with open(os.path.expanduser("~") + "/openai_key.json") as json_file:
    key = json.load(json_file)
    api_key = key["my_openai_api_key"]
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

# ----- CoT PROMPT ----
agent_name = ["Alice", "Bob"]

CoT_PROMPT = f"""You are an excellent planner who is tasked with helping 2 embodied robots named {agent_name[0]} and {agent_name[1]} carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.
They can perform the following actions: ["navigate to object <object_id>", "rotate in <rotation> direction", "pick up object <object_id>", "put object on <receptacle_id>", "open object <object_id>", "close object <object_id>", "slice object <object_id>", “toggle object <object_id> on”, “toggle object <object_id> off”, "clean object <object_id>", "look up by angle <angle>", "look down by angle <angle>", “move in <translation> direction", "stay idle", "Done"]
Here "Done" is used when all the robots have completed the main task. Only use it when you think all the subtasks are complete.
"stay idle" is used when you want the robot to stay idle for one time step. This could be used to wait for the other robot to complete its subtask. Use it only when you think it is necessary.
Here <rotation> can be one of ["Right", "Left"].
Here <angle> is the angle in degrees and can only be one of [30, 60, 90, 120, 150, 180].
Here <translation> can be one of ["Ahead", "Back", "Left", "Right”].

You need to suggest the action that each robot should take at the current time step.

### Important Notes ###
* The robots can hold only one object at a time.
For example: If {agent_name} is holding an apple, it cannot pick up another object until it puts the apple down.
* Even if the robot can see objects, it might not be able to interact with them if they are too far away. Hence you will need to make the robot navigate closer to the objects they want to interact with.
For example: An action like "pick up <object_id>" is feasible only if robot can see the object and is close enough to it. So you will have to navigate closer to it before you can pick it up.
In some scenarios, the agents might not see the objects that they want to interact with. In such cases, you will have to make the robot explore the environment to find the object.
In such scenarios you can use actions to rotate in place or look up / down or navigate to explore the environment.
* If you open an object, please ensure that you close it before you navigate to a different place.
* Opening object like drawers, cabinets, fridge can block the path of the robot. So open objects only when you think it is necessary.

### INPUT FORMAT ###
* You will get a description of the task robots are supposed to do.
* You will get an image of the environment at the current time step from {agent_name[0]}'s perspective and {agent_name[1]}'s perspective as the observation input. To help you with detecting objects in the image, you will also get a list objects each agent is able to see in the environment. Here the objects are named as "<object_name>_<object_id>". 
* You will get a trace of the steps taken by the robots and the actions they took at each time step and whether it was successful or not.

### OUTPUT FORMAT ###
You are supposed to FIRST reason through the situation logically and step by step, then suggest the action each robot is supposed to take at the current time step.
In your output, do not have any extra text or content outside of the python dictionary as below. 
Your output should ONLY be in the form of a python dictionary as shown below:
{{"reason":"Reasoning for action plan....", "{agent_name[0]}": "action to be taken by {agent_name[0]}", "{agent_name[1]}": "action to be taken by {agent_name[1]}"}}
Put all of your reasoning inside of the "reason" key of the dictonary. Do NOT put any text, spaces, or enter keys (i.e. "/n") outside of it. 

For example: If you think {agent_name[0]} should pick up an apple and {agent_name[1]} should navigate to the fridge, you will have to give the output as:
{{"reason": "since the subtask list is empty, the robots need to transport the apple to the fridge", "{agent_name[0]}": "pick up apple", "{agent_name[1]}": "navigate to fridge"}}

Let's think step by step, but make sure to put all of your reasoning inside of the "reason" key of the dictionary!

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
"""

def get_action_llm_input(env):
    """
    Returns the input to the subtask LLM
    ### INPUT FORMAT ###
    {{Task: description of the task the robots are supposed to do,
    {agent_name[i]}'s observation: list of objects the {agent_name[0]} is observing}}
    """
    # extract the agent_name's observations based on how many agents there are
    llm_input_feats = []
    for i in range(env.num_agents):
        agent_name = env.agent_names[i]
        llm_input_feats.extend([agent_name + "'s observation", ])
    return dict((k, env.input_dict[k]) for k in llm_input_feats)


def convert_obs_list_str(string_list):
    """
    Convert a string representation of a list to a list
    """
    my_list = ast.literal_eval(string_list)
    # Join all elements in the list with commas
    result = ', '.join(my_list)
    return result


def prepare_prompt(env, user_prompt:str, step_num:int, summarizer=None):
    """
    module_name: str 
    choose from planner, verifier, action
    """
    system_prompt = CoT_PROMPT
    user_prompt += f"\nStep {step_num}:\n"
    out_dict = get_action_llm_input(env)
    for i in range(env.num_agents):
        agent = env.agent_names[i]
        obs = convert_obs_list_str(out_dict[f"{agent}'s observation"])

        filtered_obs = obs
        if summarizer:
            filtered_obs = summarizer.get_gpt_response(obs)
        user_prompt += f"{agent} observes {filtered_obs}\n"
    
    return system_prompt, user_prompt

def encode_image(image_path:str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def get_action_llm_input(env):
    """
    Returns the input to the subtask LLM
    ### INPUT FORMAT ###
    {{Task: description of the task the robots are supposed to do,
    {agent_name[i]}'s observation: list of objects the {agent_name[0]} is observing}}
    """
    # extract the agent_name's observations based on how many agents there are
    llm_input_feats = ["Task"]
    for i in range(env.num_agents):
        agent_name = env.agent_names[i]
        llm_input_feats.extend([agent_name + "'s observation", ])
    return dict((k, env.input_dict[k]) for k in llm_input_feats)

def prepare_payload(env, config, user_prompt, step_num:int, summarizer=None):
    """
    # payload consists of 
    * the image from each agent's perspective
    * the system prompt (which is constant)
    * the user prompt (which changes based on the state)
    This is then sent to the openai api to get the response (action or plan or verification of the plan)
    """
    system_prompt, user_prompt = prepare_prompt(env, user_prompt, step_num, summarizer=summarizer)
    base64_image = []
    image_path = env.get_frame(0)
    base64_image.append(encode_image(image_path))
    image_path = env.get_frame(1)
    base64_image.append(encode_image(image_path))
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image[0]}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image[1]}"},
                    },
                ],
            }
        ],
        "max_tokens": 1000,
        "temperature": config.temperature,
    }
    return payload, user_prompt


def get_action(response):
    """
    Given response from GPT model, extract the dictionary of actions
    """
    response_dict = response.json()
    # convert the string to a dict
    output = response_dict["choices"][0]["message"]["content"]
    dict_match = re.search(r'\{.*\}', output)

    if dict_match:
        # extract the dictionary from the matched string
        return json.loads(dict_match.group())
    else:
        return None

def get_gpt_response(env, config, user_prompt, step_num:int, summarizer=None):
    """
    General function to get the response from the GPT model given a user prompt
    """
    payload, user_prompt = prepare_payload(env, config, user_prompt, step_num, summarizer=summarizer)
    response = requests.post(
    "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
)
    return response, user_prompt

def action_checker(env, actions):
    """
    Get closest valid action

    The action output from the model is in natural language.
    This function will find the env feasible action which has the closest embedding 
    to the natural language action output from the model.
    Eg: "pick up the apple" -> "PickupObject(Apple_1)"
    """
    checked_actions = []
    for act in actions:
        preaction=act
        act = get_closest_feasible_action(act)
        action_type = act.split("(")[0]
        if action_type in ['PickupObject', 'PutObject', 'OpenObject', 'CloseObject', 'SliceObject', 'NavigateTo', 'ToggleObjectOn', 'ToggleObjectOff', 'CleanObject']:
            act = get_closest_object_id(act, env.object_dict, preaction=preaction)
        checked_actions.append(act)
    return checked_actions

def prepare_prompt_post_action(env, user_prompt, action_text, action, action_successes):
    """
    Update the user prompt with the action taken by the agents and whether it was successful or not
    """
    for i in range(env.num_agents):
        agent = env.agent_names[i]
        user_prompt += (
            f"{agent}'s intended action was {action_text[i]}, "
            f"but it was {'successful' if action_successes[i] else 'unsuccessful'} "
            f"in executing the {action}\n"
            )

    return user_prompt


def parse_action(response):
    """
    Given response from GPT model, extract the action from the response
    """
    dict = response.json()
    output = dict["choices"][0]["message"]["content"]

    match = re.search(r'({.*})', output, re.DOTALL)
    if match:
        json_string = match.group(0)
        parse_dict = json.loads(json_string)
        return parse_dict["Action"]
    else:
        raise Exception("No match found")
