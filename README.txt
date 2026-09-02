Git 仓库地址：待创建公开仓库后填写。

运行：
1. 安装 Python 3.11+。
2. 设置 OPENAI_API_KEY；兼容网关可设置 OPENAI_BASE_URL。
3. 执行：python cli.py --root . "请检查项目并修复测试失败"

每次启动 CLI 都会显示 `Hello, NJU Software Institute!` 欢迎行，并在交互终端中以简洁状态行展示当前 agent 的定位；`--no-color` 可关闭颜色。

任务完成后 CLI 会继续显示 Task>，可以追问并复用同一 session 上下文。/sessions 列出会话，/new 新建，/switch <id> 切换，/exit 退出。每条消息原子保存到用户目录 .coding-agent/sessions；重启后使用 --continue-session 或 --session <id> 恢复。--once 保持单轮脚本语义；--session-dir/CODING_AGENT_SESSION_DIR 可改存储位置。

文件回滚命令：/checkpoints 列出当前 session 中仍可撤销的 Agent 文件编辑；/undo 撤销最近一次 write_file 或 apply_patch；/rollback 撤销最近一个发生文件编辑的任务中的全部编辑。编辑前内容和 SHA-256 校验信息默认持久化在用户目录 .coding-agent/snapshots，可用 --undo-dir 或 CODING_AGENT_UNDO_DIR 修改。若文件在 Agent 编辑后又被用户或其他程序修改，回滚会拒绝覆盖并保留现状。run_command 及其引起的 Git、依赖、数据库、网络或外部系统副作用不在文件回滚范围内。

任务完成采用显式验证闭环：Agent 修改文件后必须调用 `verify_task` 执行测试、lint 或构建命令，并且命令退出码为 0；随后调用 `finish_task` 提交完成摘要，CLI 才会把任务标记为已验证完成。若模型只输出“完成”或在验证前调用 `finish_task`，本地完成门禁会拒绝结束并要求继续处理。没有文件修改的普通问答可以直接返回。

上下文管理：`--context-tokens` 设置近似总上下文预算，`--output-tokens` 为模型回复预留空间，`--tool-result-chars` 限制活动上下文中的单个工具结果，`--task-summary-chars` 限制确定性任务摘要长度。模型请求会压缩并去重工具输出，再按完整任务轮次裁剪；被裁剪的旧任务会提取任务、文件、验证和错误事实形成摘要；完整 session 历史仍原样保存，不会因为一次裁剪而永久丢失。

默认终端只显示 Thinking 状态、工具动作摘要、编辑 diff、命令输出末尾 20 行和流式最终回答；read/search/list 正文隐藏。-v 显示只读结果和完整命令输出，-vv 另外显示原始 tool call/API payload，-q 隐藏工具摘要但保留状态与答案，--no-color 关闭 ANSI 颜色。副作用确认提示为“? Run/Write/Apply patch ...”，工具错误只显示一行红字；完整审计仍写入 --audit-log 指定的 JSONL。

特色：从零实现，不使用 LangChain、AutoGen 等 agent 框架。模型只提出工具调用，本地程序自行完成 SSE/非流式响应解析、工具执行、循环终止、上下文裁剪和 session 持久化；工具包括 list_files、read_file、read_image、read_pdf、read_docx、search_files、write_file、apply_patch、run_command、verify_task、finish_task，并带工作区路径限制和破坏性命令拦截。

文档和图片读取：`read_image` 可以把 workspace 内的 PNG/JPEG 等图片作为视觉输入交给支持 vision 的模型；`read_pdf` 按页提取 PDF 文本，传入 `include_images=true` 时同时渲染页面；无文本层的扫描 PDF 会自动尝试渲染。`read_docx` 提取 DOCX 的段落和表格，传入 `include_images=true` 时同时附加 `word/media` 中的嵌入图片。结果会保留页码/段落/表格标记并受 `max_output_chars` 限制。第一版不支持旧式二进制 `.doc`，请先转换为 `.docx` 或 PDF；图片会自动校验并在必要时压缩到受控大小。PDF 页面渲染需要 PyMuPDF 或系统 `pdftoppm`，配置的中转模型还必须支持视觉输入。

模型接口默认使用 Chat Completions；auto 模式遇到 404/405 时回退 Responses。可用 --base-url、--model、--api-mode 配置。

测试：python -m unittest discover -s working_directory/tests -v
