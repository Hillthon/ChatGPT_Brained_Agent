Git 仓库地址：待创建公开仓库后填写。

运行：
1. 安装 Python 3.11+。
2. 设置 OPENAI_API_KEY；兼容网关可设置 OPENAI_BASE_URL。
3. 执行：python cli.py --root . "请检查项目并修复测试失败"

任务完成后 CLI 会继续显示 Task>，可以追问并复用同一 session 上下文。/sessions 列出会话，/new 新建，/switch <id> 切换，/exit 退出。每条消息原子保存到用户目录 .coding-agent/sessions；重启后使用 --continue-session 或 --session <id> 恢复。--once 保持单轮脚本语义；--session-dir/CODING_AGENT_SESSION_DIR 可改存储位置。

默认终端只显示 Thinking 状态、工具动作摘要、编辑 diff、命令输出末尾 20 行和流式最终回答；read/search/list 正文隐藏。-v 显示只读结果和完整命令输出，-vv 另外显示原始 tool call/API payload，-q 隐藏工具摘要但保留状态与答案，--no-color 关闭 ANSI 颜色。副作用确认提示为“? Run/Write/Apply patch ...”，工具错误只显示一行红字；完整审计仍写入 --audit-log 指定的 JSONL。

特色：从零实现，不使用 LangChain、AutoGen 等 agent 框架。模型只提出工具调用，本地程序自行完成 SSE/非流式响应解析、工具执行、循环终止、上下文裁剪和 session 持久化；工具包括 list_files、read_file、search_files、write_file、apply_patch、run_command，并带工作区路径限制和破坏性命令拦截。

模型接口默认使用 Chat Completions；auto 模式遇到 404/405 时回退 Responses。可用 --base-url、--model、--api-mode 配置。

测试：python -m unittest discover -s working_directory/tests -v
