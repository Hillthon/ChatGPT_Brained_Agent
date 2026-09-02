# 设计说明

本项目按小步迭代实现一个 coding agent，核心逻辑全部位于本地 Python 程序中，不依赖 LangChain、LlamaIndex、OpenAI Agents SDK、Claude Agent SDK、AutoGen 或 CrewAI。

## 迭代阶段

1. **最小闭环**：`OpenAICompatibleClient` 发送 OpenAI 兼容的 Chat Completions/Responses 请求，`CodingAgent.run` 保存统一消息格式、解析 `tool_calls`，调用本地工具，直到模型返回普通文本。
2. **真实编程工具**：加入目录浏览、带行号读取、文本搜索、整文件写入、单文件 unified diff 和命令执行。
3. **安全边界**：所有路径通过 `Path.resolve()` 限制在 workspace；写入、补丁和命令逐次确认；拦截常见破坏性命令；命令超时且截断输出。
4. **可靠性**：工具异常作为 `ERROR` 观察结果回传模型；最大步骤数防止死循环；上下文按完整对话组裁剪；可选 JSONL 审计日志记录副作用。
5. **连续对话与会话**：CLI 在单个任务完成后继续接收输入；`SessionStore` 将多个独立消息历史持久化，并支持列出、新建、恢复和切换。
6. **终端交互**：agent 只产生结构化事件，Renderer 决定默认、详细和安静模式的可见内容；模型文本可通过 SSE 增量输出。
7. **持久化文件回滚**：`write_file` 和 `apply_patch` 在写入前保存文件快照，按 session 和任务记录检查点；CLI 支持单步撤销、整任务回滚和检查点查看。
8. **完成与验证闭环**：文件修改后必须经过 `verify_task` 的零退出码验证，再由 `finish_task` 显式提交摘要；本地循环拒绝未经验证的完成声明，并把拒绝原因反馈给模型继续执行。

## 两层循环与事件

```text
session -> 用户任务 -> 模型 <-> 本地工具循环 -> 最终回答
   ^         |                                  |
   |         +---------- 完整历史 --------------+
   +---- checkpoint JSON <- 继续提问 / 切换 session
```

`CodingAgent.run()` 发出 `("thinking", {...})`、`("tool_call", {...})`、`("tool_start", {...})`、`("tool_end", {...})`、`("assistant_delta", text)`、`("verification_required", {...})`、`("completion_rejected", {...})` 和 `("run_end", {...})`。模型客户端的 `model_request`、`model_response`、`model_chunk` 事件只在 `-vv` 显示。这样工具执行逻辑不依赖终端格式，CLI 也不会把裸参数、session JSON 或 API payload 混入默认输出。

模型没有本地文件系统或代码执行权限。它只能请求 `TOOL_SCHEMAS` 中列出的操作，实际副作用由 `Workspace` 完成，因此不依赖服务端 Code Interpreter 或 Files API。

## 会话边界

- `CodingAgent.run()` 只结束当前任务，不销毁 agent；下一次调用会在同一 `messages` 上追加新的 user turn。
- CLI 默认保持交互循环；`--once` 为自动化脚本保留原有单轮语义。`/new` 创建独立历史，`/switch` 恢复指定历史，`--continue-session` 在进程重启后恢复当前工作区最近的历史。
- session 采用版本化 JSON，每条 user/assistant/tool 消息后立即 checkpoint，并通过同目录临时文件加 `replace` 原子更新。文件不保存 API key，且绑定规范化 workspace 路径，避免在错误项目中恢复上下文。
- 默认 session 目录位于用户目录而非被 agent 操作的 workspace，防止模型通过文件工具读取会话控制数据；可显式覆盖存储位置。
- 超过上下文字符预算时按完整 user turn 从旧到新丢弃，工具调用与结果不会拆开；即使最新 turn 本身超预算也会保留，避免静默丢失当前请求。

## 终端输出策略

- 默认输出保持低噪音：Thinking 使用 `\r` 原地状态行；工具完成后输出 `✓/✗/⏱ + 动词 + 对象 + 状态`。只读工具正文默认不显示，命令只显示尾部 20 行。
- `write_file` 和 `apply_patch` 在批准回调之前通过 `Workspace.preview()` 生成 unified diff，Renderer 先展示 diff，再由 CLI 提示确认；这使审核看到的内容与实际执行保持一致。
- `-v` 打开只读工具结果和完整命令输出，`-vv` 加上原始工具调用及模型请求/响应/chunk；`-q` 隐藏动作行但仍保留等待状态和最终回答。错误默认一行显示，详细 traceback 只在 `-vv` 出现。
- Chat Completions SSE 会重组文本和分片 tool-call arguments；Responses 支持 `response.output_text.delta`、`response.function_call_arguments.delta` 等事件。若网关忽略 `stream` 返回普通 JSON，解析器仍接受该响应。

## 文件回滚边界

- `UndoManager` 仅跟踪由 `write_file` 和 `apply_patch` 成功产生的文件变化。每次编辑保存编辑前字节、权限位以及编辑前后 SHA-256；新建文件记录为“撤销时删除”，没有变化或被拒绝、失败的编辑不产生检查点。
- 快照默认存放在 workspace 外的 `~/.coding-agent/snapshots/<session-id>`，索引和二进制快照使用同目录临时文件原子写入。索引绑定 session ID 和规范化 workspace；进程重启后仍可恢复，但另一个 session 不能使用这些检查点。
- `/undo` 恢复当前 session 最近一次仍有效的文件编辑；`/rollback` 按逆序恢复最近一个有文件变化的任务。恢复前会比较当前文件与记录的编辑后哈希；如果用户、其他程序或后续未记录操作改过文件，则抛出冲突而不是强制覆盖。
- 这是文件级、Agent 行为级回滚，不是虚拟机或完整文件系统事务。`run_command` 可能修改任意文件、Git 状态、依赖、数据库或外部服务，这些副作用不被承诺可撤销，执行确认中会明确提示这一点。

## 完成判定与验证闭环

- 过去模型返回任意普通文本就会结束任务，无法区分“已完成”“验证失败”和“暂时无法继续”。现在本地循环维护三个简单状态：本轮是否改过文件、`verify_task` 是否成功、是否收到 `finish_task`。
- `verify_task` 是显式验证工具，复用现有命令执行和审批机制，并将退出码为 0 的结果标记为 `verification_passed`；非零退出、超时或拒绝都会标记为失败并回传模型。
- 只要本轮改过文件，普通文本不会结束任务，`finish_task` 在验证通过前也会返回错误。模型收到错误后可以修复、重新验证，再提交完成摘要。
- 没有文件修改的解释型任务仍允许普通文本结束，这是对问答场景的必要简化；真正的代码修改任务必须经过“修改 -> 验证 -> 完成提交”三步。
- 最大步数仍是最终保险丝。达到上限时返回“任务尚未确认完成”，而不是伪装成成功。

## 可辩护的取舍

- 使用标准库 HTTP 客户端而非模型 SDK，减少依赖并让请求、响应解析和错误路径清晰可见；`--base-url`、`--api-mode` 兼容 OpenAI 风格中转网关，并支持 Chat Completions 与 Responses。
- `apply_patch` 只接受单文件 unified diff，并逐行验证上下文，避免整文件重写覆盖用户改动。
- 默认不自动批准副作用，CLI 逐次询问；自动化测试注入 `approve=lambda _: True`，模型客户端也可替换为 fake client。
- 上下文裁剪采用确定性策略而非额外调用模型总结，成本和失败路径更可控；代价是很早的细节可能被丢弃，后续可增加摘要层改善长期记忆。
