from app.utils.common_utils import transform_link, split_footnotes
from app.utils.log_util import logger
import time
from app.schemas.response import (
    CoderMessage,
    WriterMessage,
    ModelerMessage,
    SystemMessage,
    CoordinatorMessage,
)
from app.services.redis_manager import redis_manager
from litellm import acompletion
import litellm
from app.schemas.enums import AgentType
from app.utils.track import agent_metrics
from icecream import ic

litellm.callbacks = [agent_metrics]

class LLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        task_id: str,
        max_tokens: int | None = None,
        reasoning_effort: str | None = "max",
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.chat_count = 0
        self.max_tokens = max_tokens
        self.task_id = task_id
        self.reasoning_effort = reasoning_effort  # DeepSeek V4 思考强度: "high" | "max"

    @staticmethod
    def message_to_dict(message) -> dict:
        """将 API 响应消息转为 dict，确保 reasoning_content 不丢失（DeepSeek V4 思考模式必需）。

        DeepSeek V4 默认开启思考模式，模型输出的 reasoning_content 在工具调用场景下
        必须在后续请求中原样回传，否则 API 返回 400。
        """
        if isinstance(message, dict):
            return message
        msg_dict = message.model_dump() if hasattr(message, 'model_dump') else dict(message)
        # 显式保留 reasoning_content（部分 litellm 版本的 model_dump 可能遗漏此字段）
        reasoning = getattr(message, 'reasoning_content', None)
        if reasoning is not None:
            msg_dict['reasoning_content'] = reasoning
        return msg_dict

    async def chat(
        self,
        history: list = None,
        tools: list = None,
        tool_choice: str = None,
        max_retries: int = 8,  # 添加最大重试次数
        retry_delay: float = 1.0,  # 添加重试延迟
        top_p: float | None = None,  # 添加top_p参数,
        response_format: dict | None = None,  # JSON Output: {"type": "json_object"}
        agent_name: AgentType = AgentType.SYSTEM,  # CoderAgent or WriterAgent
        sub_title: str | None = None,
    ) -> str:
        logger.info(f"subtitle是:{sub_title}")

        # 验证和修复工具调用完整性
        if history:
            history = self._validate_and_fix_tool_calls(history)

        kwargs = {
            "api_key": self.api_key,
            "model": self.model,
            "messages": history,
            "stream": False,
            "top_p": top_p,
            "metadata": {"agent_name": agent_name},
        }

        # DeepSeek V4 思考强度 (high / max)，通过 extra_body 透传以绕过 litellm 参数校验
        if self.reasoning_effort:
            extra = kwargs.get("extra_body", {})
            extra["reasoning_effort"] = self.reasoning_effort
            kwargs["extra_body"] = extra

        # JSON Output 模式（CoordinatorAgent / ModelerAgent）
        if response_format:
            kwargs["response_format"] = response_format

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        if self.base_url:
            kwargs["base_url"] = self.base_url
        litellm.enable_json_schema_validation = True #加入json格式验证

        # TODO: stream 输出
        for attempt in range(max_retries):
            try:
                # completion = self.client.chat.completions.create(**kwargs)
                response = await acompletion(**kwargs)
                logger.info(f"API返回: {response}")
                if not response or not hasattr(response, "choices"):
                    raise ValueError("无效的API响应")
                self.chat_count += 1
                await self.send_message(response, agent_name, sub_title)
                return response
            except Exception as e:
                logger.error(f"第{attempt + 1}次重试: {str(e)}")
                if attempt < max_retries - 1:  # 如果不是最后一次尝试
                    time.sleep(retry_delay * (attempt + 1))  # 指数退避
                    continue
                logger.debug(f"请求参数: {kwargs}")
                raise  # 如果所有重试都失败，则抛出异常

    def _validate_and_fix_tool_calls(self, history: list) -> list:
        """验证并修复工具调用完整性"""
        if not history:
            return history

        ic(f"🔍 开始验证工具调用，历史消息数量: {len(history)}")

        # 查找所有未匹配的tool_calls
        fixed_history = []
        i = 0

        while i < len(history):
            msg = history[i]

            # 如果是包含tool_calls的消息
            if isinstance(msg, dict) and "tool_calls" in msg and msg["tool_calls"]:
                ic(f"📞 发现tool_calls消息在位置 {i}")

                # 检查每个tool_call是否都有对应的response，分别处理
                valid_tool_calls = []
                invalid_tool_calls = []

                for tool_call in msg["tool_calls"]:
                    tool_call_id = tool_call.get("id")
                    ic(f"  检查tool_call_id: {tool_call_id}")

                    if tool_call_id:
                        # 查找对应的tool响应
                        found_response = False
                        for j in range(i + 1, len(history)):
                            if (
                                history[j].get("role") == "tool"
                                and history[j].get("tool_call_id") == tool_call_id
                            ):
                                ic(f"  ✅ 找到匹配响应在位置 {j}")
                                found_response = True
                                break

                        if found_response:
                            valid_tool_calls.append(tool_call)
                        else:
                            ic(f"  ❌ 未找到匹配响应: {tool_call_id}")
                            invalid_tool_calls.append(tool_call)

                # 根据检查结果处理消息
                if valid_tool_calls:
                    # 有有效的tool_calls，保留它们
                    fixed_msg = msg.copy()
                    fixed_msg["tool_calls"] = valid_tool_calls
                    fixed_history.append(fixed_msg)
                    ic(
                        f"  🔧 保留 {len(valid_tool_calls)} 个有效tool_calls，移除 {len(invalid_tool_calls)} 个无效的"
                    )
                else:
                    # 没有有效的tool_calls，移除tool_calls但可能保留其他内容
                    cleaned_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
                    if cleaned_msg.get("content"):
                        fixed_history.append(cleaned_msg)
                        ic(f"  🔧 移除所有tool_calls，保留消息内容")
                    else:
                        ic(f"  🗑️ 完全移除空的tool_calls消息")

            # 如果是tool响应消息，检查是否是孤立的
            elif isinstance(msg, dict) and msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                ic(f"🔧 检查tool响应消息: {tool_call_id}")

                # 查找对应的tool_calls
                found_call = False
                for j in range(len(fixed_history)):
                    if fixed_history[j].get("tool_calls") and any(
                        tc.get("id") == tool_call_id
                        for tc in fixed_history[j]["tool_calls"]
                    ):
                        found_call = True
                        break

                if found_call:
                    fixed_history.append(msg)
                    ic(f"  ✅ 保留有效的tool响应")
                else:
                    ic(f"  🗑️ 移除孤立的tool响应: {tool_call_id}")

            else:
                # 普通消息，直接保留
                fixed_history.append(msg)

            i += 1

        if len(fixed_history) != len(history):
            ic(f"🔧 修复完成: {len(history)} -> {len(fixed_history)} 条消息")
        else:
            ic(f"✅ 验证通过，无需修复")

        # --- DeepSeek V4 思考模式兼容：清理无效的 reasoning_content ---
        # 规则：只有携带 tool_calls 的 assistant 消息才需要保留 reasoning_content，
        # 其余消息中的 reasoning_content 应移除，避免 API 400 错误。
        for msg in fixed_history:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                has_tool_calls = bool(msg.get("tool_calls"))
                if not has_tool_calls and "reasoning_content" in msg:
                    ic(f"🧹 移除无 tool_calls 消息中的 reasoning_content")
                    del msg["reasoning_content"]

        return fixed_history

    async def send_message(self, response, agent_name, sub_title=None):
        logger.info(f"subtitle是:{sub_title}")
        content = response.choices[0].message.content

        match agent_name:
            case AgentType.CODER:
                agent_msg: CoderMessage = CoderMessage(content=content)
            case AgentType.WRITER:
                # 处理 Markdown 格式的图片语法
                content, _ = split_footnotes(content)
                content = transform_link(self.task_id, content)
                agent_msg: WriterMessage = WriterMessage(
                    content=content,
                    sub_title=sub_title,
                )
            case AgentType.MODELER:
                agent_msg: ModelerMessage = ModelerMessage(content=content)
            case AgentType.SYSTEM:
                agent_msg: SystemMessage = SystemMessage(content=content)
            case AgentType.COORDINATOR:
                agent_msg: CoordinatorMessage = CoordinatorMessage(content=content)
            case _:
                raise ValueError(f"不支持的agent类型: {agent_name}")

        await redis_manager.publish_message(
            self.task_id,
            agent_msg,
        )


# class DeepSeekModel(LLM):
#     def __init__(
#         self,
#         api_key: str,
#         model: str,
#         base_url: str,
#         task_id: str,
#     ):
#         super().__init__(api_key, model, base_url, task_id)
# self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)


async def simple_chat(model: LLM, history: list) -> str:
    """
    Description of the function.

    Args:
        model (LLM): 模型
        history (list): 构造好的历史记录（包含system_prompt,user_prompt）

    Returns:
        return_type: Description of the return value.
    """
    kwargs = {
        "api_key": model.api_key,
        "model": model.model,
        "messages": history,
        "stream": False,
    }

    if model.base_url:
        kwargs["base_url"] = model.base_url

    response = await acompletion(**kwargs)

    return response.choices[0].message.content
