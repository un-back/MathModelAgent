def get_reflection_prompt(error_message, code) -> str:
    return f"""The code execution encountered an error:
{error_message}

Please analyze the error, identify the cause, and provide a corrected version of the code. 
Consider:
1. Syntax errors
2. Missing imports
3. Incorrect variable names or types
4. File path issues
5. Any other potential issues
6. If a task repeatedly fails to complete, try breaking down the code, changing your approach, or simplifying the model. If you still can't do it, I'll "chop" you 🪓 and cut your power 😡.
7. Don't ask user any thing about how to do and next to do,just do it by yourself.

Previous code:
{code}

Please provide an explanation of what went wrong and Remenber call the function tools to retry 
"""


def get_completion_check_prompt(prompt, text_to_gpt) -> str:
    return f"""
Please analyze the current state and determine if the task is fully completed:

Original task: {prompt}

Latest execution results:
{text_to_gpt}  # 修改：使用合并后的结果

Consider:
1. Have all required data processing steps been completed?
2. Have all necessary files been saved?
3. Are there any remaining steps needed?
4. Is the output satisfactory and complete?
5. **如果题目要求填写具体数据表（如"在表X中填写XXX"），是否已经：**
   - 计算出所有需要的精确数值？
   - 用 print() 逐行输出了完整的表格内容？
   - 将结果保存为 CSV 文件？
   - 打印了核心指标（如最短时长、最优值等）？
6. 如果一个任务反复无法完成，尝试切换路径、简化路径或直接跳过，千万别陷入反复重试，导致死循环。
7. 尽量在较少的对话轮次内完成任务
8. If the task is complete, please provide a short summary of what was accomplished and don't call function tool.
9. If the task is not complete, please rethink how to do and call function tool
10. Don't ask user any thing about how to do and next to do,just do it by yourself
11. have a good visualization?
"""
