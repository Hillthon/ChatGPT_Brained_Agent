Git 仓库地址：https://github.com/Hillthon/ChatGPT_Brained_Agent

## 运行

Python 3.11+，安装依赖：

    python -m pip install pypdf pillow pymupdf

通过环境变量设置 `CODING_AGENT_API_KEY`（不要把密钥写入仓库）。启动：

    python cli.py --root . "请检查项目并修复测试失败"

中转模型可用 `CODING_AGENT_BASE_URL` 或 `--base-url`；`--once` 执行单轮任务。
启动时显示 `Hello, NJU Software Institute!`。

## 特色

项目为个人从零实现，不使用 LangChain、LlamaIndex、OpenAI Agents SDK、Claude Agent SDK、AutoGen、CrewAI 等框架，也不依赖服务端 Code Interpreter、Files API。模型仅提出工具调用，解析、工具本地执行、上下文管理、循环终止、错误处理和权限控制均由本项目实现。

支持目录浏览、文件读写、搜索、unified diff、命令、测试验证和完成提交。路径限制在 workspace；写入、补丁、命令需确认并拦截常见破坏性命令。修改后必须 `verify_task` 成功，才能 `finish_task`。

支持连续对话及多个持久化 session：`/new`、`/sessions`、`/switch <id>`、`--continue-session`；`/undo` 撤销最近编辑，`/rollback` 回滚最近任务编辑并检查外部冲突。

上下文采用输出预留、工具结果压缩去重、任务轮次裁剪和确定性摘要，完整历史仍持久保存。`read_pdf`、`read_docx` 提取文档，`read_image` 或 `include_images=true` 将图片交给 vision 模型。旧式 `.doc` 和扫描 PDF 的 OCR 暂不支持。

测试：

    python -m unittest discover -s working_directory/tests -v
