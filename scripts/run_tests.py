#!/usr/bin/env python3
"""LangDeep 测试运行脚本 — 从框架根目录启动所有测试。

用法:
    python scripts/run_tests.py                    # 运行全部
    python scripts/run_tests.py -v                 # 详细模式
    python scripts/run_tests.py --filter planner   # 运行 planner 相关测试
    python scripts/run_tests.py --list             # 列出所有模块
    python scripts/run_tests.py --failfast         # 首次失败停止
"""

import sys
import os

# 将 tests 目录加入路径，委托给 run_all.py
_tests_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
sys.path.insert(0, _tests_dir)

if __name__ == "__main__":
    from run_all import main
    main()
