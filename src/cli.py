"""Command line interface for LangDeep developer operations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from langdeep import __version__
from langdeep.core.cache.registry import cache_registry
from langdeep.core.diagnostics import build_doctor_report, validate_runtime
from langdeep.core.im.registry import im_channel_registry
from langdeep.core.memory.registry import memory_registry
from langdeep.core.observability import HealthChecker
from langdeep.core.registry.agent_registry import agent_registry
from langdeep.core.registry.model_registry import model_registry, provider_registry
from langdeep.core.registry.tool_registry import tool_registry
from langdeep.core.sandbox.registry import sandbox_registry


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Run the LangDeep CLI and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not hasattr(args, "handler"):
        parser.print_help()
        return 0

    return args.handler(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="langdeep",
        description="LangDeep runtime diagnostics and registry utilities.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"langdeep {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    health = subparsers.add_parser("health", help="Run runtime health checks.")
    health.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Maximum seconds per individual health probe.",
    )
    health.set_defaults(handler=_handle_health)

    diagnostics = subparsers.add_parser(
        "diagnostics",
        help="Validate registry wiring before serving traffic.",
    )
    diagnostics.add_argument(
        "--instantiate-agents",
        action="store_true",
        help="Instantiate registered agents while validating.",
    )
    diagnostics.set_defaults(handler=_handle_diagnostics)

    doctor = subparsers.add_parser(
        "doctor",
        help="Run environment, dependency, registry, health, and security diagnostics.",
    )
    doctor.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when warnings are present.",
    )
    doctor.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format.",
    )
    doctor.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Maximum seconds per individual health probe.",
    )
    doctor.add_argument(
        "--no-instantiate-agents",
        action="store_true",
        help="Skip agent instantiation during runtime diagnostics.",
    )
    doctor.set_defaults(handler=_handle_doctor)

    registry_list = subparsers.add_parser("list", help="List registered LangDeep components.")
    registry_list.add_argument(
        "registry",
        nargs="?",
        default="all",
        choices=("all", "models", "providers", "agents", "tools", "memory", "cache", "im", "sandbox"),
        help="Registry to list.",
    )
    registry_list.set_defaults(handler=_handle_list)

    return parser


def _handle_health(args: argparse.Namespace) -> int:
    status = HealthChecker(version=__version__).check_all(timeout=args.timeout)
    payload = _json_ready(status)
    _print_json(payload)
    return 1 if payload.get("status") == "unhealthy" else 0


def _handle_diagnostics(args: argparse.Namespace) -> int:
    diagnostics = validate_runtime(instantiate_agents=args.instantiate_agents)
    payload = diagnostics.to_dict()
    _print_json(payload)
    return 0 if diagnostics.ok else 1


def _handle_doctor(args: argparse.Namespace) -> int:
    payload = build_doctor_report(
        strict=args.strict,
        instantiate_agents=not args.no_instantiate_agents,
        timeout=args.timeout,
    )
    if args.format == "text":
        _print_doctor_text(payload)
    else:
        _print_json(payload)
    return 0 if payload["ok"] else 1


def _handle_list(args: argparse.Namespace) -> int:
    payload = _registry_snapshot()
    if args.registry != "all":
        payload = {args.registry: payload[args.registry]}
    _print_json(payload)
    return 0


def _registry_snapshot() -> Dict[str, Any]:
    return {
        "models": model_registry.list_models(),
        "providers": provider_registry.list_providers(),
        "agents": agent_registry.list_agents(),
        "tools": tool_registry.list_tools(),
        "memory": memory_registry.list_backends(),
        "cache": cache_registry.list_backends(),
        "im": im_channel_registry.list_channels(),
        "sandbox": sandbox_registry.list_backends(),
    }


def _print_json(payload: Any) -> None:
    print(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True))


def _print_doctor_text(payload: Dict[str, Any]) -> None:
    print(f"LangDeep doctor: {payload['status']}")
    print(f"Python: {payload['environment']['python']}")
    print(f"Errors: {payload['error_count']}")
    print(f"Warnings: {payload['warning_count']}")
    print("Registries:")
    for name, values in payload["registries"].items():
        print(f"- {name}: {len(values)}")
    if payload["issues"] or payload["diagnostics"]["issues"]:
        print("Issues:")
        for issue in payload["diagnostics"]["issues"] + payload["issues"]:
            print(f"- {issue['severity']} {issue['component']}/{issue['name']}: {issue['message']}")


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
