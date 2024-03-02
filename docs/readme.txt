


接受消息的函数
channel.wechat.wechat_channel.WechatChannel.handle_group

消费消息的函数
channel.chat_channel.ChatChannel.consume
    核心处理
    bot.chatgpt.chat_gpt_bot.ChatGPTBot.reply
        调用第三模型的返回数据
        bot.chatgpt.chat_gpt_bot.ChatGPTBot.reply_text
            bridge.bridge.Bridge.fetch_reply_content
                bot.chatgpt.chat_gpt_bot.ChatGPTBot.reply
                    bot.openai.open_ai_bot.OpenAIBot.reply_text