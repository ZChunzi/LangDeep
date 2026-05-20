"""Repository maintenance tests for GitHub templates."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
ISSUE_TEMPLATE_DIR = REPO_ROOT / ".github" / "ISSUE_TEMPLATE"
PR_TEMPLATE = REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
DOCS_DIR = REPO_ROOT / "docs"


def test_issue_templates_have_required_form_fields():
    template_files = sorted(
        path for path in ISSUE_TEMPLATE_DIR.glob("*.yml")
        if path.name != "config.yml"
    )

    assert {path.name for path in template_files} >= {
        "bug_report.yml",
        "feature_request.yml",
        "docs.yml",
        "good_first_issue.yml",
    }

    for template_file in template_files:
        template = yaml.safe_load(template_file.read_text(encoding="utf-8"))
        assert template["name"], template_file.name
        assert template["description"], template_file.name
        assert template["title"], template_file.name
        assert isinstance(template["labels"], list) and template["labels"], template_file.name
        assert isinstance(template["body"], list) and template["body"], template_file.name
        assert any(
            item.get("validations", {}).get("required") is True
            for item in template["body"]
        ), template_file.name

        seen_ids = set()
        for item in template["body"]:
            assert item["type"], template_file.name
            assert item["id"], template_file.name
            assert item["id"] not in seen_ids, template_file.name
            seen_ids.add(item["id"])
            assert item["attributes"]["label"], template_file.name


def test_issue_template_config_contact_links_are_complete():
    config_file = ISSUE_TEMPLATE_DIR / "config.yml"
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))

    assert config["blank_issues_enabled"] is True
    assert isinstance(config["contact_links"], list)
    assert config["contact_links"]
    for link in config["contact_links"]:
        assert link["name"]
        assert link["url"].startswith("https://")
        assert link["about"]


def test_pull_request_template_covers_review_requirements():
    template = PR_TEMPLATE.read_text(encoding="utf-8")

    required_sections = [
        "## 背景 / Summary",
        "Related issue:",
        "## 目标 / Goal",
        "## 修改内容 / Changes",
        "## 验收标准 / Acceptance criteria",
        "## 测试 / Tests",
        "## 风险 / Risks",
        "## 备注 / Notes for reviewers",
    ]
    for section in required_sections:
        assert section in template

    risk_prompts = [
        "Compatibility:",
        "Security/privacy:",
        "Performance/operations:",
        "Rollback plan:",
    ]
    for prompt in risk_prompts:
        assert prompt in template

    assert "python -m pytest" in template
    assert "python -m ruff check" in template


def test_release_documentation_is_present():
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    checklist = (REPO_ROOT / "docs" / "release-checklist.md").read_text(encoding="utf-8")

    assert "## [Unreleased]" in changelog
    assert "## [2.0.13]" in changelog
    assert "TestPyPI" in checklist
    assert "trusted publishing" in checklist
    assert "python -m twine check dist/*" in checklist
    assert "Rollback" in checklist


def test_ci_workflow_includes_coverage_artifact():
    workflow = yaml.safe_load((WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8"))

    coverage = workflow["jobs"]["coverage"]
    assert coverage["name"] == "Coverage"
    steps = coverage["steps"]
    assert any("pytest --cov=src" in step.get("run", "") for step in steps)
    assert any(step.get("uses") == "actions/upload-artifact@v4" for step in steps)


def test_publish_workflow_supports_testpypi_and_pypi_trusted_publishing():
    workflow = yaml.safe_load((WORKFLOW_DIR / "publish.yml").read_text(encoding="utf-8"))
    on_section = workflow.get("on") or workflow.get(True)
    jobs = workflow["jobs"]

    assert "workflow_dispatch" in on_section
    assert on_section["push"]["tags"] == ["v*"]
    assert {"build", "publish-testpypi", "publish-pypi"} <= set(jobs)
    assert jobs["publish-testpypi"]["permissions"]["id-token"] == "write"
    assert jobs["publish-pypi"]["permissions"]["id-token"] == "write"

    testpypi_steps = jobs["publish-testpypi"]["steps"]
    assert any(
        step.get("with", {}).get("repository-url") == "https://test.pypi.org/legacy/"
        for step in testpypi_steps
    )
    build_steps = jobs["build"]["steps"]
    assert any("twine check dist/*" in step.get("run", "") for step in build_steps)


def test_docs_index_links_core_documentation_pages():
    expected_pages = {
        "index.md",
        "getting-started.md",
        "concepts.md",
        "decorators.md",
        "orchestrator.md",
        "providers.md",
        "agents.md",
        "tools.md",
        "workflow-plan.md",
        "memory-cache.md",
        "sandbox.md",
        "observability.md",
        "security.md",
        "deployment.md",
        "api-reference.md",
        "developer-guide.md",
    }
    existing_pages = {path.name for path in DOCS_DIR.glob("*.md")}
    assert expected_pages <= existing_pages

    index_text = (DOCS_DIR / "index.md").read_text(encoding="utf-8")
    for page in expected_pages - {"index.md"}:
        assert f"({page})" in index_text

    readme_en = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    readme_zh = (REPO_ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    for page in ("getting-started.md", "concepts.md", "deployment.md", "api-reference.md"):
        assert f"docs/{page}" in readme_en
        assert f"docs/{page}" in readme_zh
