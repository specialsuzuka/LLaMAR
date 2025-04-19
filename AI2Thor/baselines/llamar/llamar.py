# 导入必要的库和模块
import re
import base64
import requests
import json, os
import pandas as pd
import tqdm
from pathlib import Path
import sys
import openai  # 确保安装了 openai 库：pip install openai



#===========================================================
CALLCOUNT = 0
#===========================================================


# 设置当前目录为工作目录，便于处理相对路径
directory = Path(os.getcwd()).absolute()
sys.path.append(str(directory))  # 添加当前目录到 Python 路径中，便于导入模块
print(os.getcwd())

# 导入 AI2Thor 环境模块
from AI2Thor.env_new import AI2ThorEnv
# 解析命令行参数
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=int, default=0)  # 任务 ID
parser.add_argument("--floorplan", type=int, default=0)  # 楼层平面图 ID
parser.add_argument("--verbose", action="store_true")  # 是否打印详细信息
parser.add_argument("--action_verbose", action="store_true")  # 是否打印动作详细信息
parser.add_argument("--name", type=str, default="llamar")  # 基线名称
parser.add_argument("--agents", type=int, default=2)  # 代理数量
parser.add_argument("--config_file", type=str, default="config.json")  # 配置文件路径
args = parser.parse_args()

# 创建并写入配置文件
with open("AI2Thor/baselines/llamar/config.json", "w+") as f:
    d = {
        "num_agents": args.agents,
        "tasks": [
            {
                "task_id": args.task,
                "floorplan": args.floorplan,
                # 添加其他任务相关配置
            }
        ],
    }
    json.dump(d, f)

# 导入工具函数和配置类
from AI2Thor.baselines.llamar.llamar_utils_multiagent import *
from AI2Thor.baselines.utils import Logger, AutoConfig

import warnings
import os

# 禁用并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore")

# 加载 OpenAI API 密钥
with open("./openai_key.json") as json_file:
    key = json.load(json_file)
    api_key = key["my_openai_api_key"]
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

# 自动配置任务参数
auto = AutoConfig(config_file=args.config_file)
auto.set_task(args.task)
auto.set_floorplan(args.floorplan)
auto.set_agents(args.agents)
timeout = auto.get_task_timeout()  # 获取任务超时时间

# 初始化环境
config = auto.config()
env = AI2ThorEnv(config)
env.verbose = True  # 禁用环境的详细输出
d = env.reset(task=auto.task_string())  # 重置环境
# d = env.step(
#     [
#         "Move(Ahead)",
#     ]
# )
logger = Logger(env=env, baseline_name=args.name)  # 初始化日志记录器
print("baseline path (w/ results):", logger.baseline_path)

# 初始化变量
previous_action = [] * config.num_agents  # 记录上一步的动作
previous_success = [True, True]  # 初始化为成功状态

# 打印基线启动信息
print("*" * 50)
print("Starting the llamar baseline")
print("*" * 50)

# PLANNER - 初始规划器
success = False
while not success:
    try:
        # 调用 GPT 模型生成初始计划
        CALLCOUNT+=1
        response = get_gpt_response(env, config, action_or_planner="planner",CALLCOUNT = CALLCOUNT)
        outdict = get_action(response)  # 解析 GPT 响应
        if args.verbose:
            print("Planner Output:\n", outdict)
        success = True
    except Exception as e:
        print("failure reason (in try-except loop):", e)
        pass

# 主循环：执行任务直到完成或超时
for step_num in tqdm.trange(timeout):

    # 更新计划，基于当前环境状态和已完成的子任务
    update_plan(env, outdict["plan"], env.closed_subtasks)
    # ACTOR - 执行动作
    success = False
    while not success:
        try:
            # 调用 GPT 模型生成动作
            CALLCOUNT+=1
            response = get_gpt_response(env, config, action_or_planner="action",CALLCOUNT = CALLCOUNT)
            # print("调用一次动作VLM")
            outdict = get_action(response)  # 解析动作响应
            success = True
            if args.verbose or args.action_verbose:
                print("*" * 10, "Actor outdict!", "*" * 10)
                print(outdict)
                print()
        except:
            pass

    # 处理动作输出，提取动作、原因、子任务、记忆等信息
    preaction, reason, subtask, memory, failure_reason = process_action_llm_output(
        outdict
    )
    action = print_stuff(env, preaction, reason, subtask, memory, failure_reason)

    # 记录代理的记忆信息
    logger.log_agent_mem(env.step_num, action, reason, subtask, memory)

    # 更新共享子任务和记忆（如果启用）
    if config.use_shared_subtask:
        env.update_subtask(subtask, 0)
    if config.use_shared_memory:
        env.update_memory(memory, 0)

    # 执行动作并获取结果
    d1, successes = env.step(action)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
    # video_writer = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1280, 720))
    # frame = d1.cv2img  # 获取BGR格式图像[3](@ref)
    # video_writer.write(frame)
    previous_action = action
    previous_success = successes

    # 打印相关信息（如果启用详细模式）
    if args.verbose:
        print_relevant_info(env, config, env.input_dict)

    # VERIFIER - 验证动作结果
    success = False
    while not success:
        try:
            # 调用 GPT 模型验证动作
            CALLCOUNT+=1
            response = get_gpt_response(env, config, action_or_planner="verifier",CALLCOUNT = CALLCOUNT)
            outdict = get_action(response)
            if args.verbose:
                print("Verifier Output:\n", outdict)
            success = True
        except:
            pass

    # 更新已完成的子任务列表
    env.closed_subtasks = outdict["completed subtasks"]
    if len(env.closed_subtasks) == 0:
        env.closed_subtasks = None
    env.input_dict["Robots' completed subtasks"] = env.closed_subtasks
    env.get_planner_llm_input()

    # PLANNER - 更新计划
    success = False
    while not success:
        try:
            CALLCOUNT+=1
            response = get_gpt_response(env, config, action_or_planner="planner",CALLCOUNT = CALLCOUNT)
            outdict = get_action(response)
            success = True
        except:
            pass

    # 获取统计信息并完成当前步骤
    coverage = env.checker.get_coverage()
    transport_rate = env.checker.get_transport_rate()
    finished = env.checker.check_success()

    # 记录当前步骤的日志
    logger.log_step(
        step=step_num,
        preaction=preaction,
        action=action,
        success=previous_success,
        coverage=coverage,
        transport_rate=transport_rate,
        finished=finished,
    )

    # 打印步骤信息（如果启用详细模式）
    if args.verbose:
        print("_" * 50)
        print(f"Step {step_num}")
        print(f"Completed Subtasks: ")
        print("\n".join(env.checker.subtasks_completed))

    # 如果所有代理都输出 "Done"，则结束循环
    if all(status == "Done" for status in action):
        break

# 停止环境控制器
env.controller.stop()
