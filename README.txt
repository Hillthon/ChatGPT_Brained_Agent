Git 仓库地址：待创建公开仓库后填写。

运行：
1. 安装 Python 3.11+。
2. 设置环境变量 OPENAI_API_KEY（也可设置 OPENAI_BASE_URL 使用 OpenAI 兼容网关）。
3. 在项目目录执行：python cli.py --root . "请检查项目并修复测试失败"。
   写文件、应用补丁和执行命令前会逐次询问确认；可加 --audit-log audit.jsonl 记录 JSONL 审计日志。

特色：这是从零实现的、无 LangChain/AutoGen 等 agent 框架的 coding agent。模型只提出工具调用，本地程序自行解析并执行 list_files、read_file、search_files、write_file、apply_patch、run_command；包含工作区路径沙箱、破坏性命令拦截、操作确认、超时、输出截断、错误回传、最大步数和上下文裁剪。工具循环会在模型不再提出调用时结束，并输出总结。

设计分阶段：先实现“模型+工具+循环”的最小闭环，再加入安全策略、上下文管理和可注入的模型客户端，测试无需网络或密钥即可运行。API key 默认从环境变量读取，不写入仓库；CLI 也支持显式传入。

演示建议：准备一个有失败测试的小项目，要求 agent 读取代码、修改实现并运行测试，全程展示确认提示和最终结果。

中转站兼容配置：
- 默认地址为 `https://rightapi.ai/codex/v1`，也可用 `--base-url`、`CODING_AGENT_BASE_URL`、`RIGHTAPI_BASE_URL` 或 `OPENAI_BASE_URL` 覆盖。
- 密钥优先从 `CODING_AGENT_API_KEY`、`RIGHTAPI_API_KEY`、`OPENAI_API_KEY` 读取，也可通过 `--api-key` 传入。
- `--api-mode auto` 会优先请求 `/chat/completions`；若中转站返回 404/405，则自动尝试 `/responses`。可用 `--api-mode chat` 或 `--api-mode responses` 固定协议。
- 示例：`python cli.py --root . --model gpt-4o-mini "请检查项目"`
