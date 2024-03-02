# encoding:utf-8

import time

import json
from openai import OpenAI
import openai._exceptions
import requests
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionFunctionMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionToolMessageParam, \
    ChatCompletionMessageToolCallParam

from bot.bot import Bot
from bot.chatgpt.chat_gpt_session import ChatGPTSession
from bot.openai.open_ai_image import OpenAIImage
from bot.session_manager import SessionManager
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from common.token_bucket import TokenBucket
from config import conf, load_config


# OpenAI对话模型API (可用)
class ChatGPTBot(Bot, OpenAIImage):
    def __init__(self):
        super().__init__()
        # set the default api_key
        openai.api_key = conf().get("open_ai_api_key")
        if conf().get("open_ai_api_base"):
            openai.api_base = conf().get("open_ai_api_base")

        self.client = OpenAI(api_key=conf().get("open_ai_api_key"), base_url=conf().get("open_ai_api_base"))
        proxy = conf().get("proxy")
        if proxy:
            openai.proxy = proxy
        if conf().get("rate_limit_chatgpt"):
            self.tb4chatgpt = TokenBucket(conf().get("rate_limit_chatgpt", 20))

        self.sessions = SessionManager(ChatGPTSession, model=conf().get("model") or "gpt-3.5-turbo")
        self.args = {
            "model": conf().get("model") or "gpt-3.5-turbo",  # 对话模型的名称
            "temperature": conf().get("temperature", 0.9),  # 值在[0,1]之间，越大表示回复越具有不确定性
            # "max_tokens":4096,  # 回复最大的字符数
            "top_p": conf().get("top_p", 1),
            "frequency_penalty": conf().get("frequency_penalty", 0.0),  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            "presence_penalty": conf().get("presence_penalty", 0.0),  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            # "request_timeout": conf().get("request_timeout", None),  # 请求超时时间，openai接口默认设置为600，对于难问题一般需要较长时间
            "timeout": conf().get("request_timeout", None),  # 重试超时时间，在这个时间内，将会自动重试
            "tools": [],
            "tool_choice": "auto"
        }
        self.tool_choice_plagin_init()

    def tool_choice_plagin_init(self):
        self.tool_choice_plagin_dh_qy_init()

    def tool_choice_plagin_dh_qy_init(self):
        json = {
            "name": "exec_daohang_qy",
            "description": "根据提供的信息执行配送中心(导航上线)迁移函数,信息一般内容为：开始进行数据迁移 向日葵：648538836/1C9yfH \n014 C22084 泸州叙永配送中心 sysdb sa/a1111111\n12112,其中014 C22084是dest_db, sysdb是src_db,a1111111是src_pwd,sa是src_uname,12112是src_port;如果你识别不准确,可以向用户二次确认",
            "parameters": {
                "type": "object",
                "properties": {
                    "src_db": {
                        "type": "string",
                        "description": "这是源数据库的名字,例如: sysdb,sysdb_LC,SysDb,sysdb01,sysdb01,sysdb_nj,sysdb_ls 等等"
                    },
                    "src_port": {
                        "type": "string",
                        "description": "这是源数据库的端口号，例如:12001,12002,12107,12109,12119,12399,12492等等"
                    },
                    "src_uname": {
                        "type": "string",
                        "description": "这是访问数据库的用户名字，例如:sa,sp,sa_ls,sa_bj 等等"
                    },
                    "src_pwd": {
                        "type": "string",
                        "description": "这是访问数据库的密码，例如:a1111111,Tuaj`rau!tus@meKoj3xav4fuck5kuv,12003,ts046ts046等等"
                    },
                    "dest_db": {
                        "type": "string",
                        "description": "这是目标数据库的名字,注意中间不要有空格，如果有空格用-替换,用小写标识出来即可，例如:014-c12001,029-c91929等等"
                    }
                },
                "required": [
                    "src_db",
                    "src_port",
                    "src_uname",
                    "src_pwd",
                    "dest_db"
                ]
            }
        }
        tool_function = {"type": "function", "function": json}
        self.args["tools"].append(tool_function)

    def reply(self, query, context=None):
        # acquire reply content
        if context.type == ContextType.TEXT:
            logger.info("[CHATGPT] query={}".format(query))

            session_id = context["session_id"]
            reply = None
            clear_memory_commands = conf().get("clear_memory_commands", ["#清除记忆"])
            if query in clear_memory_commands:
                self.sessions.clear_session(session_id)
                reply = Reply(ReplyType.INFO, "记忆已清除")
            elif query == "#清除所有":
                self.sessions.clear_all_session()
                reply = Reply(ReplyType.INFO, "所有人记忆已清除")
            elif query == "#更新配置":
                load_config()
                reply = Reply(ReplyType.INFO, "配置已更新")
            if reply:
                return reply
            session = self.sessions.session_query(query, session_id)
            logger.debug("[CHATGPT] session query={}".format(session.messages))

            api_key = context.get("openai_api_key")
            model = context.get("gpt_model")
            new_args = None
            if model:
                new_args = self.args.copy()
                new_args["model"] = model
            # if context.get('stream'):
            #     # reply in stream
            #     return self.reply_text_stream(query, new_query, session_id)

            reply_content = self.reply_text(session, api_key, args=new_args)
            logger.debug(
                "[CHATGPT] new_query={}, session_id={}, reply_cont={}, completion_tokens={}".format(
                    session.messages,
                    session_id,
                    reply_content["content"],
                    reply_content["completion_tokens"],
                )
            )
            if reply_content["completion_tokens"] == 0 and len(reply_content["content"]) > 0:
                reply = Reply(ReplyType.ERROR, reply_content["content"])
            elif reply_content["completion_tokens"] > 0:
                self.sessions.session_reply(reply_content["content"], session_id, reply_content["total_tokens"])
                reply = Reply(ReplyType.TEXT, reply_content["content"])
            else:
                reply = Reply(ReplyType.ERROR, reply_content["content"])
                logger.debug("[CHATGPT] reply {} used 0 tokens.".format(reply_content))
            return reply

        elif context.type == ContextType.IMAGE_CREATE:
            ok, retstring = self.create_img(query, 0)
            reply = None
            if ok:
                reply = Reply(ReplyType.IMAGE_URL, retstring)
            else:
                reply = Reply(ReplyType.ERROR, retstring)
            return reply
        else:
            reply = Reply(ReplyType.ERROR, "Bot不支持处理{}类型的消息".format(context.type))
            return reply

    def session_to_list(self, messages):
        logger.debug("messages".format(messages))

        rs_arr = []
        for m in messages:
            if m["role"] in "user":
                rs_arr.append(ChatCompletionUserMessageParam(content=m["content"], role=m["role"]))
            if m["role"] in "system":
                rs_arr.append(ChatCompletionSystemMessageParam(content=m["content"], role=m["role"]))
            if m["role"] in "assistant":
                rs_arr.append(ChatCompletionAssistantMessageParam(content=m["content"], role=m["role"]))
            if m["role"] in "tool":
                rs_arr.append(ChatCompletionToolMessageParam(content=m["content"], role=m["role"]))
            if m["role"] in "function":
                rs_arr.append(ChatCompletionFunctionMessageParam(content=m["content"], role=m["role"]))
        return rs_arr

    def do_function_choice(self, tool_calls, messgaes, **args):
        while (tool_calls is not None):
            tool_calls_arr = []
            for tool_call in tool_calls:
                tool_calls_arr.append(
                    ChatCompletionMessageToolCallParam(id=tool_call.id, function=tool_call.function, type="function"))
            messgaes.append(
                ChatCompletionAssistantMessageParam(role="assistant", tool_calls=tool_calls_arr))

            # 1106 版新模型支持一次返回多个函数调用请求
            for tool_call in tool_calls:
                arguments = json.loads(tool_call.function.arguments)
                logger.debug("收到 args: {}".format(arguments))
                if tool_call.function.name == "exec_daohang_qy":
                    print("Call: exec_daohang_qy")
                    result = "开始迁移中: exec_daohang_qy"

                print("=====函数返回=====")
                messgaes.append(
                    ChatCompletionToolMessageParam(content=str(result), role="tool", tool_call_id=tool_call.id))
            response = self.client.chat.completions.create(messages=messgaes, **args)
            if hasattr(response, "content") and response.content is None:  # 解决 OpenAI 的一个 400 bug
                response.content = ""

            # 把大模型的回复加入到对话中
            # messgaes.append(
            #     ChatCompletionAssistantMessageParam(role="assistant", content=response.choices[0].message))
            return response

    def reply_text(self, session: ChatGPTSession, api_key=None, args=None, retry_count=0) -> dict:
        """
        call openai's ChatCompletion to get the answer
        :param session: a conversation session
        :param session_id: session id
        :param retry_count: retry count
        :return: {}
        """
        try:
            if conf().get("rate_limit_chatgpt") and not self.tb4chatgpt.get_token():
                raise openai.RateLimitError("RateLimitError: rate limit exceeded")
            # if api_key == None, the default openai.api_key will be used
            if args is None:
                args = self.args
            messgaes = self.session_to_list(session.messages)
            response = self.client.chat.completions.create(messages=messgaes, **args)
            logger.debug("[CHATGPT] response={}".format(response))
            # logger.info("[ChatGPT] reply={}, total_tokens={}".format(response.choices[0]['message']['content'], response["usage"]["total_tokens"]))

            # todo 针对funs 进行编写
            if response.choices is not None:
                response_inner = self.do_function_choice(response.choices[0].message.tool_calls, messgaes, **args)
                if response_inner is not None:
                    response = response_inner

            return {
                "total_tokens": response.usage.total_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "content": response.choices[0].message.content,
            }
        except Exception as e:
            need_retry = retry_count < 2
            result = {"completion_tokens": 0, "content": "我现在有点累了，等会再来吧"}
            if isinstance(e, openai.RateLimitError):
                logger.warn("[CHATGPT] RateLimitError: {}".format(e))
                result["content"] = "提问太快啦，请休息一下再问我吧"
                if need_retry:
                    time.sleep(20)
            elif isinstance(e, openai.Timeout):
                logger.warn("[CHATGPT] Timeout: {}".format(e))
                result["content"] = "我没有收到你的消息"
                if need_retry:
                    time.sleep(5)
            elif isinstance(e, openai.APIError):
                logger.warn("[CHATGPT] Bad Gateway: {}".format(e))
                result["content"] = "请再问我一次"
                if need_retry:
                    time.sleep(10)
            elif isinstance(e, openai.APIConnectionError):
                logger.warn("[CHATGPT] APIConnectionError: {}".format(e))
                result["content"] = "我连接不到你的网络"
                if need_retry:
                    time.sleep(5)
            else:
                logger.exception("[CHATGPT] Exception: {}".format(e))
                need_retry = False
                self.sessions.clear_session(session.session_id)

            if need_retry:
                logger.warn("[CHATGPT] 第{}次重试".format(retry_count + 1))
                return self.reply_text(session, api_key, args, retry_count + 1)
            else:
                return result


class AzureChatGPTBot(ChatGPTBot):
    def __init__(self):
        super().__init__()
        openai.api_type = "azure"
        openai.api_version = conf().get("azure_api_version", "2023-06-01-preview")
        self.args["deployment_id"] = conf().get("azure_deployment_id")

    def create_img(self, query, retry_count=0, api_key=None):
        api_version = "2022-08-03-preview"
        url = "{}dalle/text-to-image?api-version={}".format(openai.api_base, api_version)
        api_key = api_key or openai.api_key
        headers = {"api-key": api_key, "Content-Type": "application/json"}
        try:
            body = {"caption": query, "resolution": conf().get("image_create_size", "256x256")}
            submission = requests.post(url, headers=headers, json=body)
            operation_location = submission.headers["Operation-Location"]
            retry_after = submission.headers["Retry-after"]
            status = ""
            image_url = ""
            while status != "Succeeded":
                logger.info("waiting for image create..., " + status + ",retry after " + retry_after + " seconds")
                time.sleep(int(retry_after))
                response = requests.get(operation_location, headers=headers)
                status = response.json()["status"]
            image_url = response.json()["result"]["contentUrl"]
            return True, image_url
        except Exception as e:
            logger.error("create image error: {}".format(e))
            return False, "图片生成失败"
