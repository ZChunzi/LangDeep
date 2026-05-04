# Contributing to LangDeep

感谢您对 LangDeep 的关注！我们欢迎各种形式的贡献。

## 反馈与讨论

- **Bug 报告 / 功能建议**：请在 GitHub Issues 提交
- **安全相关问题**：请直接邮件联系维护者，不要公开提交 Issue

## 开发流程

1. Fork 本仓库
2. 创建您的功能分支：`git checkout -b feat/my-feature`
3. 提交您的改动：`git commit -m 'feat: add some feature'`
4. 推送到分支：`git push origin feat/my-feature`
5. 提交 Pull Request

## 环境搭建

```bash
git clone https://github.com/ZChunzi/langdeep.git
cd langdeep/LangDeep
pip install -e ".[all]"
```

## 代码规范

- Python 版本 >= 3.9
- 遵循 [PEP 8](https://peps.python.org/pep-0008/) 编码风格
- 提交信息使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：
  - `feat:` 新功能
  - `fix:` Bug 修复
  - `docs:` 文档变更
  - `refactor:` 重构
  - `test:` 测试相关
  - `chore:` 构建/工具链变更

## Pull Request 指南

- 确保 PR 描述清楚改动目的和实现方式
- 新功能应包含对应测试用例
- 确保所有测试通过：`python scripts/run_tests.py`
- 保持 PR 范围聚焦，避免无关改动

## 测试指南

测试框架位于 `LangDeep/tests/`，使用自定义运行器（非 pytest）。

### 运行测试

```bash
cd LangDeep

# 运行全部 247 个测试
python scripts/run_tests.py

# 详细输出
python scripts/run_tests.py -v

# 按模块筛选（支持模块名或测试名）
python scripts/run_tests.py --filter executor

# 列出所有测试模块
python scripts/run_tests.py --list

# 首次失败即停止
python scripts/run_tests.py --failfast
```

### 测试组织

```
tests/
├── conftest.py                  # 共享夹具: SmartMockLLM, clean_registries, orch()
├── run_all.py                   # 统一运行器（按依赖顺序加载模块）
├── __init__.py                  # 包标记
├── test_errors.py               # 异常层次结构 (10)
├── test_logging.py              # 日志与追踪 (7)
├── test_tool_registry.py        # 工具注册中心 (7)
├── test_agent_registry.py       # Agent 注册中心 (8)
├── test_model_registry.py       # 模型注册中心 (11)
├── test_execution_policy.py     # 执行策略 (8)
├── test_decorators.py           # 注解装饰器 (5)
├── test_prompt_loader.py        # Prompt 加载器 (11)
├── test_clean_messages.py       # 消息清理 (11)
├── test_keyword_routing.py      # 关键字路由 (14)
├── test_agent_node.py           # Agent 节点 (10)
├── test_retry_task_runner.py    # 重试任务运行器 (11)
├── test_planner.py              # 规划器与模板 (19)
├── test_aggregator.py           # 聚合器与 Merger (18)
├── test_executor.py             # 执行器基础 (10)
├── test_executor_edge.py        # 执行器边界: 循环依赖、并发、部分失败 (10)
├── test_workflow_planner.py     # 工作流规划器 (13)
├── test_task_scheduler.py       # 定时调度器 (16)
├── test_orchestrator.py         # 编排器基础 (14)
├── test_orchestrator_edge.py    # 编排器边界: 扩展点注入、异常流 (8)
├── test_fuzz.py                 # 模糊测试: 随机输入不变式 (4)
└── test_agent_capabilities.py   # 端到端集成测试 (24)
```

### 编写测试

- 每个测试文件需要定义 `setup_function()`，在每条测试前调用 `clean_registries()` 重置单例
- 使用 `conftest.py` 中的 `orch()` 工厂创建 `FlowOrchestrator` 实例
- 使用 `SmartMockLLM` 模拟 LLM 行为（无需真实 API）
- 异步测试使用 `asyncio.run(run())` 模式，无需 pytest 插件
- 新增测试文件需要在 `run_all.py` 的 `TEST_MODULES` 列表中注册（按依赖顺序）
- 测试应尽量验证行为而非实现细节，优先使用公共 API

### 新增测试文件步骤

1. 在 `tests/` 下创建 `test_*.py`
2. 定义 `setup_function()` 调用 `clean_registries()`
3. 编写以 `test_` 开头的测试函数
4. 在 `run_all.py` 的 `TEST_MODULES` 中注册（注意依赖顺序）
5. 运行 `python scripts/run_tests.py --filter your_module` 验证

## 许可证

贡献即表示您同意您的贡献基于 [MIT](./LICENSE) 许可证授权。
