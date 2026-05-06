#!/usr/bin/env python3
"""
LangDeep 测试运行器 — 一键运行所有单元测试和集成测试。

用法:
    python tests/run_all.py                    # 运行所有测试
    python tests/run_all.py -v                 # 详细输出
    python tests/run_all.py --filter keyword   # 只运行文件名或测试名包含 keyword 的
    python tests/run_all.py --list             # 列出所有测试模块
    python tests/run_all.py --failfast         # 首次失败即停止
"""

import argparse
import importlib
import os
import sys
import time
import traceback
from pathlib import Path

# Ensure project root and tests dir are on path
_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, os.path.abspath(_project_root))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Test discovery ────────────────────────────────────────────────────

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

# Ordered by dependency (simpler modules first)
TEST_MODULES = [
    "test_errors",
    "test_logging",
    "test_tool_registry",
    "test_execution_policy",
    "test_agent_registry",
    "test_model_registry",
    "test_decorators",
    "test_prompt_loader",
    "test_clean_messages",
    "test_keyword_routing",
    "test_agent_node",
    "test_retry_task_runner",
    "test_planner",
    "test_aggregator",
    "test_executor",
    "test_executor_edge",
    "test_workflow_planner",
    "test_task_scheduler",
    "test_orchestrator",
    "test_orchestrator_edge",
    # New modules
    "test_memory",
    "test_cache",
    "test_im",
    # Integration tests last
    "test_fuzz",
    "test_agent_capabilities",
]


def discover_tests(filter_str: str = "") -> list:
    """Return list of test module names matching filter."""
    results = []
    for mod in TEST_MODULES:
        if filter_str and filter_str.lower() not in mod.lower():
            continue
        results.append(mod)
    return results


# ── Runner ────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0
ERROR = 0
RESULTS: list[dict] = []


def run_test_module(module_name: str, verbosity: int = 1) -> bool:
    """Run all functions starting with 'test_' in the given module.

    Returns True if all passed, False otherwise.
    """
    global PASS, FAIL, ERROR

    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        print(f"  💥 IMPORT FAILED: {module_name} — {exc}")
        traceback.print_exc()
        ERROR += 1
        RESULTS.append({"module": module_name, "status": "IMPORT_FAIL", "tests": 0, "passed": 0, "time": 0})
        return False

    # Find all test functions
    test_fns = []
    for attr_name in dir(mod):
        if attr_name.startswith("test_") and callable(getattr(mod, attr_name)):
            test_fns.append(attr_name)

    if not test_fns:
        if verbosity > 0:
            print(f"  ⚠️  {module_name}: no test_ functions found")
        return True

    passed = 0
    failed = 0
    module_start = time.perf_counter()

    setup_fn = getattr(mod, "setup_function", None)
    teardown_fn = getattr(mod, "teardown_function", None)

    for fn_name in test_fns:
        fn = getattr(mod, fn_name)
        test_start = time.perf_counter()
        try:
            if setup_fn:
                setup_fn()
            fn()
            elapsed = time.perf_counter() - test_start
            if verbosity > 0:
                print(f"  ✅ {module_name}.{fn_name} ({elapsed:.2f}s)")
            passed += 1
        except Exception as exc:
            elapsed = time.perf_counter() - test_start
            if verbosity > 0:
                print(f"  ❌ {module_name}.{fn_name} ({elapsed:.2f}s): {exc}")
            if verbosity > 1:
                traceback.print_exc()
            failed += 1
        finally:
            if teardown_fn:
                teardown_fn()

    module_elapsed = time.perf_counter() - module_start
    status = "PASS" if failed == 0 else "FAIL"
    RESULTS.append({
        "module": module_name,
        "status": status,
        "tests": len(test_fns),
        "passed": passed,
        "failed": failed,
        "time": module_elapsed,
    })
    PASS += passed
    FAIL += failed

    if failed > 0 and args.failfast:
        print(f"\n  ⛔ Fail-fast: stopping after failures in {module_name}")
        return False

    return failed == 0


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LangDeep 测试运行器")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="详细输出 (-vv 含 traceback)")
    parser.add_argument("--filter", type=str, default="", help="只运行文件名/测试名包含指定字符串的测试")
    parser.add_argument("--list", action="store_true", help="列出所有测试模块")
    parser.add_argument("--failfast", action="store_true", help="首次失败即停止")
    global args
    args = parser.parse_args()

    if args.list:
        print("可用测试模块:")
        for mod in TEST_MODULES:
            print(f"  - {mod}")
        return

    modules = discover_tests(args.filter)
    if not modules:
        print(f"未找到匹配 '{args.filter}' 的测试模块")
        sys.exit(1)

    # Filter out integration test from unit test count
    unit_modules = [m for m in modules if m != "test_agent_capabilities"]

    print("=" * 60)
    print(f"  LangDeep 测试套件")
    print(f"  模块: {len(modules)} ({len(unit_modules)} 单元 + {'1' if 'test_agent_capabilities' in modules else '0'} 集成)")
    print(f"  筛选: {'无' if not args.filter else args.filter}")
    print("=" * 60)
    print()

    global PASS, FAIL, ERROR
    start = time.perf_counter()

    for module_name in modules:
        if args.verbose:
            print(f"\n── {module_name} ──")
        ok = run_test_module(module_name, verbosity=args.verbose)
        if not ok and args.failfast:
            break

    total = time.perf_counter() - start

    # ── Summary ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  测试结果汇总")
    print("=" * 60)
    for r in RESULTS:
        icon = "✅" if r["status"] == "PASS" else "❌" if r["status"] == "FAIL" else "💥"
        print(f"  {icon} {r['module']}: {r['passed']}/{r['tests']} passed ({r['time']:.2f}s)")
    print(f"\n{'─' * 60}")
    total_n = PASS + FAIL
    print(f"  总测试: {total_n}  |  通过: {PASS} ✅  |  失败: {FAIL} ❌  |  导入错误: {ERROR}")
    print(f"  总耗时: {total:.2f}s")
    print(f"{'─' * 60}\n")

    sys.exit(0 if FAIL == 0 and ERROR == 0 else 1)


if __name__ == "__main__":
    main()
