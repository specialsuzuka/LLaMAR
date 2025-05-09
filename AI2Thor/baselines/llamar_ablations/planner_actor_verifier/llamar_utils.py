import re
import base64
import requests
import json, os
import pandas as pd
from pathlib import Path
import sys

# set parent directory to address relative imports
directory = Path(os.getcwd()).absolute()
sys.path.append(
    str(directory)
)  # note: no ".parent" addition is needed for python (.py) files

from AI2Thor.env_new import AI2ThorEnv
from AI2Thor.base_env import convert_dict_to_string
from AI2Thor.object_actions import get_closest_feasible_action, get_closest_object_id

with open(os.path.expanduser("~") + "/openai_key.json") as json_file:
    key = json.load(json_file)
    api_key = key["my_openai_api_key"]
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

# planner
# v0 - add additional example of cleaning in plans - avoiding extraneous actions
agent_name = ["Alice", "Bob"]
PLANNER_PROMPT = f"""You are an excellent planner who is tasked with helping 2 embodied robots named {agent_name[0]} and {agent_name[1]} carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.

You will get a description of the task robots are supposed to do. You will get an image of the environment from {agent_name[0]}'s perspective and {agent_name[1]}'s perspective as the observation input. To help you with detecting objects in the image, you will also get a list objects each agent is able to see in the environment. Here the objects are named as "<object_name>_<object_id>". 
So, along with the image inputs you will get the following information:
### INPUT FORMAT ###
{{Task: description of the task the robots are supposed to do, 
{agent_name[0]}'s observation: list of objects the {agent_name[0]} is observing,
{agent_name[1]}'s observation: list of objects the {agent_name[1]} is observing,
Robots' open subtasks: list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.
Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
Robots' combined memory: description of robots' combined memory}}

Reason over the robots' task, image inputs, observations, open subtasks, completed subtasks and memory, and then output the following:
* Reason: The reason for why new subtasks need to be added.
* Subtasks: A list of open subtasks the robots are supposed to take to complete the task. Remember, as you get new information about the environment, you can modify this list. You can keep the same plan if you think it is still valid. Do not include the subtasks that have already been completed.
The "Plan" should be in a list format where the actions are listed sequentially.
For example:
    ["locate the apple", "transport the apple to the fridge", "transport the book to the table", ""]
    ["locate the cup", "go to cup", "clean cup"]
When possible do not perform additional steps when one is sufficient (e.g. CleanObject is sufficient to clean an object, no other actions need to be done)
Your output should be in the form of a python dictionary as shown below.
Example output: {{"reason": "since the subtask list is empty, the robots need to transport the apple to the fridge and transport the book to the table", "plan": ["transport the apple to the fridge", "transport the book to the table"]}}

Ensure that the subtasks are not generic statements like "explore the environment" or "do the task". They should be specific to the task at hand.
Do not assign subtasks to any particular robot. Try not to modify the subtasks that already exist in the open subtasks list. Rather add new subtasks to the list.

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
Let's work this out in a step by step way to be sure we have the right answer.
"""

# verifier
# moves subtasks from open to completed
# v0 - add statement to encourage it append rather than overwrite completed subtasks
agent_name = ["Alice", "Bob"]
VERIFIER_PROMPT = f"""You are an excellent planner who is tasked with helping 2 embodied robots named {agent_name[0]} and {agent_name[1]} carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.

You will get a description of the task robots are supposed to do. You will get an image of the environment from {agent_name[0]}'s perspective and {agent_name[1]}'s perspective as the observation input. To help you with detecting objects in the image, you will also get a list objects each agent is able to see in the environment. Here the objects are named as "<object_name>_<object_id>".
So, along with the image inputs you will get the following information:
### INPUT FORMAT ###
{{Task: description of the task the robots are supposed to do,
{agent_name[0]}'s observation: list of objects the {agent_name[0]} is observing,
{agent_name[0]}'s state: description of {agent_name[0]}'s state,
{agent_name[0]}'s previous action: the action {agent_name[0]} took in the previous step and if it was successful,
{agent_name[1]}'s observation: list of objects the {agent_name[1]} is observing,
{agent_name[0]}'s state: description of {agent_name[0]}'s state,
{agent_name[1]}'s previous action: the action {agent_name[1]} took in the previous step,
Robots' open subtasks: list of open subtasks the robots in the previous step. If no plan has been already created, this will be None.
Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
Robots' combined memory: description of robots' combined memory}}

Reason over the robots' task, image inputs, observations, previous actions, open subtasks, completed subtasks and memory, and then output the following:
* Reason: The reason for why you think a particular subtask should be moved from the open subtasks list to the completed subtasks list.
* Completed Subtasks: The list of subtasks that have been completed by the robots. Note that you can add subtasks to this list only if they have been successfully completed and were in the open subtask list. If no subtasks have been completed at the current step, return an empty list.
The "Completed Subtasks" should be in a list format where the completed subtasks are listed. For example: ["locate the apple", "transport the apple to the fridge", "transport the book to the table"]}}
Your output should be in the form of a python dictionary as shown below.
Example output: {{"reason": "Alice placed the apple in the fridge in the previous step and was successful and Bob picked up the the book from the table. Hence Alice has completed the subtask of transporting the apple to the fridge, Bob has picked up the book, but Bob has still not completed the subtask of transporting the book to the table", "completed subtasks": ["picked up book from the table", "transport the apple to the fridge"]}}

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
When you output the completed subtasks, make sure to not forget to include the previous ones in addition to the new ones.
Let's work this out in a step by step way to be sure we have the right answer.
"""


# action planner
# v0 - change done part of the prompt (not finishing)
# v1 - add to "excellent planner" phrase "and robot controller"
agent_name = ["Alice", "Bob"]
ACTION_PROMPT = f"""You are an excellent planner and robot controller who is tasked with helping 2 embodied robots named {agent_name[0]} and {agent_name[1]} carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.
They can perform the following actions: ["navigate to object <object_id>", "rotate in <rotation> direction", "pick up object <object_id>", "put object on <receptacle_id>", "open object <object_id>", "close object <object_id>", "slice object <object_id>", “toggle object <object_id> on”, “toggle object <object_id> off”, "clean object <object_id>", "look up by angle <angle>", "look down by angle <angle>", “move in <translation> direction", "stay idle", "Done"]
Here "Done" is used when all the robots have completed the main task. Make sure to always say "Done" when you finish with the task.
"stay idle" is used when you want the robot to stay idle for one time step. This could be used to wait for the other robot to complete its subtask. Use it only when you think it is necessary.
Here <rotation> can be one of ["Right", "Left"].
Here <angle> is the angle in degrees and can only be one of [30, 60, 90, 120, 150, 180].
Here <translation> can be one of ["Ahead", "Back", "Left", "Right”].

You need to suggest the action that each robot should take at the current time step.

You will get a description of the task robots are supposed to do. You will get an image of the environment from {agent_name[0]}'s perspective and {agent_name[1]}'s perspective as the observation input.
To help you with detecting objects in the image, you will also get a list objects each agent is able to see in the environment. Here the objects are named as "<object_name>_<object_id>".
So, along with the image inputs you will get the following information:
### INPUT FORMAT ###
{{Task: description of the task the robots are supposed to do,
{agent_name[0]}'s observation: list of objects the {agent_name[0]} is observing,
{agent_name[0]}'s state: description of {agent_name[0]}'s state,
{agent_name[0]}'s previous action: description of what {agent_name[0]} did in the previous time step and whether it was successful,
{agent_name[1]}'s observation: list of objects the {agent_name[1]} is observing,
{agent_name[1]}'s state: description of {agent_name[1]}'s state,
{agent_name[1]}'s previous action: description of what {agent_name[1]} did in the previous time step and whether it was successful,
Robots' open subtasks: list of subtasks  supposed to carry out to finish the task. If no plan has been already created, this will be None.
Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
Robots' subtask: description of the subtasks the robots were trying to complete in the previous step,
Robots' combined memory: description of robot's combined memory}}

First of all you are supposed to reason over the image inputs, the robots' observations, previous actions, previous memory, subtasks and the available actions the robots can perform, and think step by step and then output the following things:
* Memory: Whatever important information about the scene you think you should remember for the future as a memory. Remember that this memory will be used in future steps to carry out the task. So, you should not include information that is not relevant to the task. You can also include information that is already present in its memory if you think it might be useful in the future.
* Reason: The reasoning for what each robot is supposed to do next
* Subtask: The subtask each robot should currently try to solve, choose this from the list of open subtasks.
* Action: The actions the robots are supposed to take just in the next step such that they make progress towards completing the task. Make sure that this suggested actions make these robots more efficient in completing the task as compared only one agent solving the task.
Your output should just be in the form of a python dictionary as shown below.
Examples of output:

Note that the output should just be a dictionary similar to the example outputs.

### Important Notes ###
* The robots can hold only one object at a time.
For example: If {agent_name} is holding an apple, it cannot pick up another object until it puts the apple down.
* Even if the robot can see objects, it might not be able to interact with them if they are too far away. Hence you will need to make the robot move closer to the objects they want to interact with.
For example: An action like "pick up <object_id>" is feasible only if robot can see the object and is close enough to it. So you will have to move closer to it before you can pick it up.
So if a particular action fails, you will have to choose a different action for the robot.
* If you open an object, please ensure that you close it before you move to a different place.
* Opening object like drawers, cabinets, fridge can block the path of the robot. So open objects only when you think it is necessary.
* When possible do not perform extraneous actions when one action is sufficient (e.g. only do CleanObject to clean an object, nothing else)

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
Let's work this out in a step by step way to be sure we have the right answer.
"""


def process_action_llm_output(outdict):
    action = []
    action.append(outdict[f"{agent_name[0]}'s action"])
    action.append(outdict[f"{agent_name[1]}'s action"])
    reason = outdict["reason"]
    subtask = outdict["subtask"]
    memory = outdict["memory"]
    return action, reason, subtask, memory


def print_stuff(env, action, reason, subtask, memory, pprint=False):
    action = action_checker(env, action)
    if pprint:
        print("Step: " + str(env.step_num))
        print(f"Action: {action}")
        print(f"Closest Action: {action}")
        print(f"Reason: {reason}")
        print(f"Subtask: {subtask}")
        print(f"Memory: {memory}")
    return action


# previous_action = []*config.num_agents
# previous_success = [True, True] # initialize with True


def encode_image(image_path: str):
    # if not os.path.exists()
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def prepare_prompt(env, module_name: str, addendum: str):
    """module_name: str
    choose from planner, verifier, action
    """
    # Choose the appropriate prompt based on what module is being called
    # user_prompt = convert_dict_to_string(env.input_dict)
    if module_name == "action":
        system_prompt = ACTION_PROMPT
        user_prompt = convert_dict_to_string(env.get_action_llm_input())
    elif module_name == "planner":
        system_prompt = PLANNER_PROMPT
        user_prompt = convert_dict_to_string(env.get_planner_llm_input())
    elif module_name == "verifier":
        system_prompt = VERIFIER_PROMPT
        user_prompt = convert_dict_to_string(env.get_verifier_llm_input())
    user_prompt += addendum
    return system_prompt, user_prompt


def prepare_payload(env, config, module_name: str, addendum: str = ""):
    """# payload consists of
    * the image from each agent's perspective
    * the system prompt (which is constant)
    * the user prompt (which changes based on the state)
    This is then sent to the openai api to get the response (action or plan or verification of the plan)
    """
    system_prompt, user_prompt = prepare_prompt(env, module_name, addendum)
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
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image[0]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image[1]}"
                        },
                    },
                ],
            },
        ],
        "max_tokens": 1000,
        "temperature": config.temperature,
    }
    return payload


def get_action(response):
    response_dict = response.json()
    # convert the string to a dict
    # json_acceptable_string = response_dict["choices"][0]["message"]["content"].replace("'", "\"").replace("\n", "").replace("json", "").replace("`", "")
    output = response_dict["choices"][0]["message"]["content"]
    json_match = re.search(r"```json(.*?)```", output, re.DOTALL)
    python_match = re.search(r"```python(.*?)```", output, re.DOTALL)
    tilde_match = re.search(r"```(.*?)```", output, re.DOTALL)
    if json_match:
        # print("Got JSON TYPE OUTPUT")
        json_data = json_match.group(1)
    elif python_match:
        # print("Got JSON TYPE OUTPUT")
        json_data = python_match.group(1)
    elif tilde_match:
        # print("Got JSON TYPE OUTPUT")
        json_data = tilde_match.group(1)
    else:
        # print("Got NORMAL TYPE OUTPUT")
        json_data = output
    # print(json_data)
    out_dict = json.loads(json_data)
    return out_dict


def get_gpt_response(env, config, action_or_planner: str, addendum: str = ""):
    payload = prepare_payload(env, config, action_or_planner, addendum)
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    return response


def print_relevant_info(env, config, input_dict):
    print("Step: " + str(env.step_num))
    print(f"{agent_name[0]}")
    print("State: " + input_dict[f"{agent_name[0]}'s state"])
    print("Previous action: " + input_dict[f"{agent_name[0]}'s previous action"])
    print(f"{agent_name[1]}")
    print("State: " + input_dict[f"{agent_name[1]}'s state"])
    print("Previous action: " + input_dict[f"{agent_name[1]}'s previous action"])
    if config.use_shared_subtask:
        print("Subtask: ", input_dict[f"Robots' subtasks"])
    elif config.use_separate_subtask:
        print(f"{agent_name[0]}'s subtask: " + input_dict[f"{agent_name[0]}'s subtask"])
        print(f"{agent_name[1]}'s subtask: " + input_dict[f"{agent_name[1]}'s subtask"])
    if config.use_shared_memory:
        print("Memory: " + input_dict["Robots' combined memory"])
    elif config.use_separate_memory:
        print(f"{agent_name[0]}'s memory: " + input_dict[f"{agent_name[0]}'s memory"])
        print(f"{agent_name[1]}'s memory: " + input_dict[f"{agent_name[1]}'s memory"])


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

        # possible actions are:
        # ["navigate to object <object_id>", "rotate in <rotation> direction", "pick up object <object_id>", "put object on <receptacle_id>", "open object <object_id>", "close object <object_id>", "slice object <object_id>", “toggle object <object_id> on”, “toggle object <object_id> off”, "clean object <object_id>", "look up by angle <angle>", "look down by angle <angle>", “move in <translation> direction", "stay idle", "Done"]
        if action_type in [
            "PickupObject",
            "PutObject",
            "OpenObject",
            "CloseObject",
            "ToggleObjectOn",
            "ToggleObjectOff",
            "SliceObject",
            "NavigateTo",
            "CleanObject",
        ]:
            act = get_closest_object_id(act, env.object_dict, preaction=preaction)
        checked_actions.append(act)
    return checked_actions


# TODO: see when we use it, don't forget to add previous_action, previous_success to the env
def check_repeat_actions(env, actions, previous_action, previous_success):
    agent_name = env.agent_names
    repeated_action = False
    repeat_str = ""
    repeat_agent_names = []  # to store names of agents who repeated actions
    for i in range(len(actions)):
        if actions[i] == previous_action[i]:
            repeated_action = True
            success_str = "successful" if previous_success[i] else "unsuccessful"
            repeat_str += f"\nYou predicted {agent_name[i]}'s action as {actions[i]} which is the same as the previous action {agent_name[i]} took. {agent_name[i]} was {success_str} in carrying out that action."
            repeat_agent_names.append(agent_name[i])

    if repeated_action:
        reason_str = f"\nPlease reason over the image inputs and the observations to suggest a different action for "
        for i in range(len(repeat_agent_names)):
            if i == 0:
                reason_str += f"{repeat_agent_names[i]}"
            else:
                reason_str += f" and {repeat_agent_names[i]}"
        reason_str += "."
        repeat_str += reason_str

    return repeated_action, repeat_str


# v0 - change it so that it appends instead
# for verifier
def set_addition(l1, l2):
    if l1 is None:
        l1 = []
    if l2 is None:
        l2 = []
    return list(set(l1 + l2))


def update_plan(env, open_subtasks, completed_subtasks):
    env.open_subtasks = open_subtasks
    env.closed_subtasks = completed_subtasks
    env.input_dict["Robots' open subtasks"] = env.open_subtasks
    env.input_dict["Robots' completed subtasks"] = env.closed_subtasks


# OLD - deprecated
def append_row(df, step, action, reason, subtask, memory):
    row = pd.DataFrame(
        [[step, action, reason, subtask, memory]],
        columns=["Step", "Action", "Reason", "Subtask", "Memory"],
    )
    df = pd.concat([df, row])
    return df
