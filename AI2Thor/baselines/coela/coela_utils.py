import re
import base64
import requests
import json, os
import pandas as pd
import random
from typing import List, Tuple, Dict
from pathlib import Path
import sys

# set directory to top folder level to address relative imports
directory = Path(os.getcwd()).absolute()
sys.path.append(str(directory)) # note: no ".parent" addition is needed for python (.py) files

# import environment
from AI2Thor.env_new import AI2ThorEnv
from AI2Thor.object_actions import (
    get_all_available_actions,
    inverse_act2text,
    inverse_subtask2text,
)

with open(os.path.expanduser("~") + "/openai_key.json") as json_file:
    key = json.load(json_file)
    api_key = key["my_openai_api_key"]
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}


def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class LLM:
    def __init__(self):
        pass

    def prepare_payload(self, system_prompt: str, user_prompt: str) -> Dict:
        """# payload consists of
        * the image from each agent's perspective
        * the system prompt (which is constant)
        * the user prompt (which changes based on the state)
        This is then sent to the openai api to get the response (action or plan or verification of the plan)
        """
        base64_image = []

        image_path = self.env.get_frame(self.agent_id)
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
                    ],
                },
            ],
            "max_tokens": 1000,
            "temperature": self.env.args.temperature,
        }
        return payload

    def get_gpt_response(self, system_prompt: str, user_prompt: str):
        payload = self.prepare_payload(system_prompt, user_prompt)
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        return response


def get_action(response):
    response_dict = response.json()
    # convert the string to a dict
    output = response_dict["choices"][0]["message"]["content"]
    dict_match = re.search(r"\{.*\}", output)

    if dict_match:
        # Extract the dictionary from the matched string
        return json.loads(dict_match.group())
    else:
        return None


class CoELA_Agent(LLM):
    def __init__(
        self,
        env,
        agent_id: int,
        agent_name: str,
        friend_name: str,
        task: int,
        total_num_steps: int,
    ):
        super().__init__()
        self.env = env
        self.agent_id = agent_id
        self.name = agent_name
        self.friend_name = friend_name
        self.task = task
        self.dialogues = [
            "Hi, I’ll let you know if I find any target objects and containers, finish any subgoals, and ask for your help when necessary.",
            "Thanks! I’ll let you know if I find any target objects and containers, finish any subgoals, and ask for your help when necessary.",
        ]
        self.dialogues_speaker = ["Alice", "Bob"]
        self.total_num_steps = total_num_steps
        random.seed(0)

    def create_agent_planner_prompt(self) -> str:
        agent_prompt = f"""I’m {self.name}. My friend {self.friend_name} and I want to complete a task together within 3000 steps. I can hold one object at a time. Given an image from my point of view of the environment, our shared goal, dialogue history, my progress, and previous actions, please help me choose the best available action to achieve the goal as soon as possible. All objects are denoted as "<object_name>_<object_id>", such as "Table_2". Actions take several steps to finish. Your output should just be the action name."""
        return agent_prompt

    def create_agent_comm_prompt(self) -> str:
        agent_prompt = f"""I’m {self.name}. My friend {self.friend_name} and I want to complete a task together within 3000 steps. I can hold one object at a time. Given an image from my point of view of the environment, our shared goal, dialogue history, my progress, and previous actions, please help me generate a short message to send to {self.friend_name} to help us achieve the goal as soon as possible. All objects are denoted as "<object_name>_<object_id>", such as "Table_2". Actions take several steps to finish."""
        return agent_prompt

    def add_goal_desc(self) -> str:
        agent_instruction = f"\nGoal: {self.task}\n"
        return agent_instruction

    def add_available_actions_prompt(self, prompt: str, message: str) -> str:
        agent_actions_prompt = (
            "Available actions: (You can only choose the action in the list)\n"
        )
        if message is not None:
            agent_actions_prompt += f"1. Send a message: {message}\n"

        inventory = self._convert_inventory()
        all_available_actions = get_all_available_actions(
            self.env.all_obs_dict[self.name], inventory
        )
        self.available_actions = all_available_actions

        for i, act in enumerate(all_available_actions):
            if message is None:
                agent_actions_prompt += f"{i+1}. {act}\n"
            elif i == len(all_available_actions) - 1:
                agent_actions_prompt += f"{i+2}. {act}\n"
                agent_actions_prompt += f"{i+3}. Done\n Done is used when all the robots have completed the main task. Only use it when you think all the subtasks are complete.\n"
                agent_actions_prompt += f"{i+4}. Stay Idle\n Stay idle is used when you want the robot to stay idle for one time step. This could be used to wait for the other robot to complete its subtask. Use it only when you think it is necessary.\n"
            else:
                agent_actions_prompt += f"{i+2}. {act}\n"
        return prompt + agent_actions_prompt

    def add_prev_actions_prompt(self, prompt: str) -> str:
        """
        prompt: str
            Prompt constructed so far
        action_history: List
            List of actions taken so far
            Example: ["NavigateTo(Bread_1)", "PickupObject(Bread_1)"]
        step_nums: List
            Example: [1, 13]
        """
        agent_prev_actions_prompt = "Previous actions: "
        if len(self.env.action_history[self.name]) == 0:
            agent_prev_actions_prompt += "No actions taken yet.\n"
            return prompt + agent_prev_actions_prompt
        assert len(self.env.action_history[self.name]) == len(
            self.env.step_nums_history[self.name]
        )
        for i, action in enumerate(self.env.action_history[self.name]):
            act_text = inverse_act2text(action)
            success_str = (
                "was successful"
                if self.env.action_success_history[self.name][i]
                else "failed"
            )
            agent_prev_actions_prompt += f"{act_text} at step {self.env.step_nums_history[self.name][i]} which {success_str}, "
        # remove the last comma and space
        agent_prev_actions_prompt = agent_prev_actions_prompt[:-2] + "\n"
        return prompt + agent_prev_actions_prompt

    def add_dialogue_history_prompt(self, prompt: str) -> str:
        """
        prompt: str
            Prompt constructed so far
        dialogues: List[str]
            List of dialogues
            Example: ["Hi, I’ll let you know if I find any target objects and containers, finish any subgoals, and ask for your help when necessary.", "Thanks! I’ll let you know if I find any target objects and containers, finish any subgoals, and ask for your help when necessary."]
        dialogues_speaker: List[str]
            Example: ["Alice", "Bob"]
        """
        agent_dialogue_history_prompt = "Dialogue history:\n"
        for i, dialogue in enumerate(self.dialogues):
            agent_dialogue_history_prompt += (
                f'{self.dialogues_speaker[i]}: "{dialogue}"\n'
            )
        return prompt + agent_dialogue_history_prompt

    def add_progress_desc(self, prompt: str) -> str:
        """This needs a checker function to get the list of completed subtasks which are pre-defined"""
        inventory = self._convert_inventory()
        progress_prompt = f"Progress: I have taken {self.env.step_num[self.agent_id]}/{self.total_num_steps}. "
        object_hold_str = (
            "I’m holding nothing. "
            if len(inventory) == 0
            else f"I’m holding {', '.join(inventory)}. "
        )
        progress_prompt += object_hold_str
        if len(self.env.checker.subtasks_completed_numerated) > 0:
            progress_prompt += f"We have successfully "
            for subtask in self.env.checker.subtasks_completed_numerated:
                # if ',' in subtask the two_objs = True
                two_objs = "," in subtask
                progress_prompt += f"{inverse_subtask2text(subtask, two_objs)}, "
        else:
            progress_prompt += "We haven't made any progress towards our goal yet. "
        return prompt + progress_prompt + "\n"

    def prepare_planner_prompt(self, message: str) -> str:
        """CoELA prompt consists of following parts:
        1. Agent prompt     [-]
        2. Goal description [-]
        3. Progress         [-]
        4. Dialogue history [-]
        5. Previous actions [-]
        6. Available actions[-]
        """
        system_prompt = self.create_agent_planner_prompt()
        prompt = self.add_goal_desc()
        prompt = self.add_progress_desc(prompt)
        prompt = self.add_dialogue_history_prompt(prompt)
        prompt = self.add_prev_actions_prompt(prompt)
        prompt = self.add_available_actions_prompt(prompt, message)
        prompt = prompt + "Answer: Let’s think step by step.\n"
        return system_prompt, prompt

    def prepare_comm_prompt(self, message: str) -> str:
        """CoELA prompt consists of following parts:
        1. Agent prompt     [-]
        2. Goal description [-]
        3. Progress         [-]
        4. Dialogue history [-]
        5. Previous actions [-]
        6. Available actions[-]
        """
        system_prompt = self.create_agent_comm_prompt()
        prompt = self.add_goal_desc()
        prompt = self.add_progress_desc(prompt)
        prompt = self.add_dialogue_history_prompt(prompt)
        prompt = self.add_prev_actions_prompt(prompt)
        prompt = self.add_available_actions_prompt(prompt, message)
        prompt = (
            prompt
            + "Note: The generated message should be accurate, helpful and brief. Do not generate repetitive messages.\n"
        )
        return system_prompt, prompt

    def _convert_inventory(self):
        inventory = self.env.inventory[self.agent_id]
        if inventory == "nothing":
            inventory = []
        else:
            inventory = [inventory]
        return inventory

    def process_action(self, act_text: str) -> Tuple[str, bool]:
        """The LLM output might contain a lot of other information that
        we don't need. This function extracts the action from the LLM output."""

        if "send a message" in act_text.lower():
            message = re.search(r"send a message: (.*)", act_text, re.IGNORECASE).group(1)
            return message, True, "send a message"
        elif "done" in act_text.lower():
            return "Done", False, "done"
        else:
            for act in self.available_actions:
                if act.lower() in act_text.lower():
                    return act, False, "action"
        # if it doesn't match any action and if it is not a message,
        # then choose a random action as done in the paper implementation
        # https://github.com/UMass-Foundation-Model/Co-LLM-Agents/blob/aea1c2e384b5d3a17908135ba1c64fcb70526d62/tdw_mat/LLM/LLM.py#L298
        # Example hallucinations: "PlaceObjectInRefrigerator(Fridge_1)"
        return random.choice(self.available_actions), False, "random"
