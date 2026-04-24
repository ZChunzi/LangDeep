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
- 确保所有测试通过：`python -m pytest tests/`
- 保持 PR 范围聚焦，避免无关改动

## 许可证

贡献即表示您同意您的贡献基于 [MIT](./LICENSE) 许可证授权。
