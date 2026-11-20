"""The files a user installs from agree with each other: one version, pinned requirements inside the allowed ranges."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

import echoview

if sys.version_info >= (3, 11):
    import tomllib
else:  # pytest brings tomli on Python 3.10
    import tomli as tomllib

REPOSITORY = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((REPOSITORY / "pyproject.toml").read_text())["project"]
# where the pins were tested: Linux, Python 3.12
LINUX = {"platform_system": "Linux", "platform_machine": "x86_64", "sys_platform": "linux", "python_version": "3.12",
         "python_full_version": "3.12.0"}


def pins(name: str) -> dict[str, Version]:
    """The `package==version` lines of a requirements file, by lower-case package name."""
    found = {}
    for line in (REPOSITORY / name).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            package, version = line.split("==")  # a line without an exact version fails here
            found[package.lower().replace("_", "-")] = Version(version)
    return found


def needed(requirements: list[str]) -> list[Requirement]:
    wanted = [Requirement(text) for text in requirements]
    return [r for r in wanted if r.marker is None or r.marker.evaluate(LINUX)]


def test_the_version_is_the_same_everywhere():
    version = PROJECT["version"]
    assert echoview.__version__ == version
    citation = (REPOSITORY / "CITATION.cff").read_text()
    assert re.search(rf"^version: {re.escape(version)}$", citation, re.M)
    newest = re.search(r"^## \[([^\]]+)\] - (\d{4}-\d{2}-\d{2})$", (REPOSITORY / "CHANGELOG.md").read_text(), re.M)
    assert newest and newest.group(1) == version
    assert re.search(rf'^date-released: "{newest.group(2)}"$', citation, re.M)


def test_requirements_pin_every_dependency_inside_its_allowed_range():
    pinned = pins("requirements.txt")
    wanted = needed(PROJECT["dependencies"] + PROJECT["optional-dependencies"]["evaluation"])
    assert {r.name.lower().replace("_", "-") for r in wanted} == set(pinned)  # nothing missing, nothing the package does not ask for
    for requirement in wanted:
        version = pinned[requirement.name.lower().replace("_", "-")]
        assert requirement.specifier.contains(version), f"{requirement} does not allow the pinned {version}"


def test_pillow_is_a_version_without_the_known_decoder_vulnerabilities():
    """Pillow before 13.0 has published vulnerabilities in decoders this package can reach with a file from outside."""
    patched = Version("13.0.0")
    assert pins("requirements.txt")["pillow"] >= patched
    asked = [Requirement(text) for text in PROJECT["dependencies"] if Requirement(text).name.lower() == "pillow"]
    newer = [r for r in asked if r.marker is None or r.marker.evaluate({**LINUX, "python_version": "3.11", "python_full_version": "3.11.0"})]
    assert len(newer) == 1 and not newer[0].specifier.contains(Version("12.3.0")) and newer[0].specifier.contains(patched)
    older = [r for r in asked if r.marker is not None and r.marker.evaluate({**LINUX, "python_version": "3.10", "python_full_version": "3.10.0"})]
    assert len(older) == 1  # Pillow 13 does not exist for Python 3.10: the package still installs there
    assert "Pillow" in (REPOSITORY / "SECURITY.md").read_text()


def test_the_developer_requirements_add_the_tools_to_the_pinned_ones():
    text = (REPOSITORY / "requirements-dev.txt").read_text()
    assert re.search(r"^-r requirements\.txt$", text, re.M)
    pinned = pins("requirements-dev.txt")
    runtime = set(pins("requirements.txt"))
    tools = [r for r in needed(PROJECT["optional-dependencies"]["dev"]) if r.name.lower().replace("_", "-") not in runtime]
    assert {r.name.lower() for r in tools} == set(pinned)
    for requirement in tools:
        assert requirement.specifier.contains(pinned[requirement.name.lower()]), str(requirement)


def test_the_image_is_built_from_the_pinned_requirements():
    dockerfile = (REPOSITORY / "Dockerfile").read_text()
    assert "requirements.txt" in dockerfile and "USER echoview" in dockerfile
    python = re.search(r"^FROM python:(\d+\.\d+)-slim", dockerfile, re.M).group(1)
    assert f"Python {python}" in (REPOSITORY / "requirements.txt").read_text()  # the pins name the Python they were tested with
    ignored = (REPOSITORY / ".dockerignore").read_text().split()
    assert ".git" in ignored and "**/*.pt" in ignored and "**/*.safetensors" in ignored  # no history and no weights in an image
