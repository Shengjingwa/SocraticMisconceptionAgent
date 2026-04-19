# SocraticMisconceptionTutor Code Wiki

本 Wiki 面向“读代码/做复现/做二次开发”的需求，按 **架构 → 模块 → API → 数据与产物 → 运行方式** 的顺序组织。

详细文档存放于 `docs/code-wiki/` 目录下：

- [1. 项目总览](docs/code-wiki/01_overview.md)
- [2. 整体架构](docs/code-wiki/02_architecture.md)
- [3. 模块说明](docs/code-wiki/03_modules.md)
- [4. 关键类与函数](docs/code-wiki/04_key_apis.md)
- [5. 数据、日志与评估产物](docs/code-wiki/05_data_outputs.md)
- [6. 运行与复现](docs/code-wiki/06_running.md)
- [7. 依赖与调用关系图](docs/code-wiki/07_dependency_graph.md)

## 项目结构
```text
src/             核心逻辑（工作流编排、分类、FSM、生成、护栏、仿真、评估）
data/            静态数据（迷思概念库、知识块、学生画像）
docs/            文档，主要包含代码百科 (code-wiki/)
README.md        项目说明、环境配置与运行方式
CODE_WIKI.md     代码百科入口
requirements.txt Python 依赖清单
```

运行时会按需生成未纳入版本控制的输出目录，如 `logs/`、`results/`、`experiments/`。
