# 设计说明

本项目按小步迭代实现一个 coding agent，核心逻辑全部位于本地 Python 程序中，不依赖 LangChain、LlamaIndex、OpenAI Agents SDK、Claude Agent SDK、AutoGen 或 CrewAI。

## 迭代阶段

1. **最小闭环**：`OpenAICompatibleClient` 发送 OpenAI 兼容的 Chat Completions/Responses 请求，`CodingAgent.run` 保存统一消息格式、解析 `tool_calls`，调用本地工具，直到模型返回普通文本。
2. **真实编程工具**：加入目录浏览、带行号读取、文本搜索、整文件写入、单文件 unified diff 和命令执行。
3. **安全边界**：所有路径通过 `Path.resolve()` 限制在 workspace；写入、补丁和命令逐次确认；拦截常见破坏性命令；命令超时且截断输出。
4. **可靠性**：工具异常作为 `ERROR` 观察结果回传模型；最大步骤数防止死循环；上下文按完整对话组裁剪；可选 JSONL 审计日志记录副作用。

## 一轮工具调用

```text
用户任务 -> system/user messages -> 模型
                       | 普通文本
                       v
                    最终回答
                       |
                       | tool_calls
                       v
          本地解析 JSON -> Workspace.execute
                       |
             tool 结果/错误 -> messages -> 模型
```

模型没有本地文件系统或代码执行权限。它只能请求 `TOOL_SCHEMAS` 中列出的操作，实际副作用由 `Workspace` 完成，因此不依赖服务端 Code Interpreter 或 Files API。

## 可辩护的取舍

- 使用标准库 HTTP 客户端而非模型 SDK，减少依赖并让请求、响应解析和错误路径清晰可见；`--base-url`、`--api-mode` 兼容 OpenAI 风格中转网关，并支持 Chat Completions 与 Responses。
- `apply_patch` 只接受单文件 unified diff，并逐行验证上下文，避免整文件重写覆盖用户改动。
- 默认不自动批准副作用，CLI 逐次询问；自动化测试注入 `approve=lambda _: True`，模型客户端也可替换为 fake client。
- 上下文裁剪优先保留最近完整工具轮次，避免把 assistant 的工具调用和 tool 结果拆开造成协议错误。
