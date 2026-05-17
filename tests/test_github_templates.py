"""Repository maintenance tests for GitHub templates."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
ISSUE_TEMPLATE_DIR = REPO_ROOT / ".github" / "ISSUE_TEMPLATE"


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
