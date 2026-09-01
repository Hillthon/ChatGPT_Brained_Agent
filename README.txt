Git 仓库地址：待创建公开仓库后填写。

运行：
1. 安装 Python 3.11+。
2. 设置环境变量 OPENAI_API_KEY；使用兼容网关时可设置 OPENAI_BASE_URL。
3. 执行：python cli.py --root . "请检查项目并修复测试失败"

任务完成后不会退出，而会继续显示 Task>，可直接追问或布置下一个任务，模型会复用本 session 的上文。会话命令：/sessions 列出、/new 新建、/switch <id> 切换、/exit 退出。每条用户、模型和工具消息都会原子保存到用户目录 .coding-agent/sessions；重启后用 --continue-session 恢复最近会话，或用 --session <id> 指定会话。可用 CODING_AGENT_SESSION_DIR/--session-dir 改位置；脚本调用可加 --once 保持单轮执行。

特色：从零实现，不使用 LangChain、AutoGen 等 agent 框架。程序自行完成模型请求、tool calling 解析、本地执行和循环终止；工具包括 list_files、read_file、search_files、write_file、apply_patch、run_command。写入、补丁和命令逐次确认，并包含工作区路径限制、破坏性命令拦截、命令超时、输出截断、错误回传、最大步数、审计日志和按完整用户轮次裁剪的上下文。session 与工作区绑定，API key 不写入会话或仓库。

模型接口：默认兼容 Chat Completions，并在 auto 模式遇到 404/405 时回退 Responses；可用 --base-url、--model、--api-mode 配置。

测试：python -m unittest discover -s working_directory/tests -v
