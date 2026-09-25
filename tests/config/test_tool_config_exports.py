"""Tool config classes re-exported by ``nanobot.config.schema`` resolve in any import order."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("first", [
    "nanobot.agent.tools.image_generation",
    "nanobot.agent.context",
    "nanobot.config.schema",
])
def test_tool_config_reexports_survive_any_import_order(first: str) -> None:
    code = (
        f"import {first}\n"
        "from nanobot.config.schema import (\n"
        "    ExecToolConfig, ImageGenerationToolConfig, ToolsConfig, WebSearchConfig,\n"
        ")\n"
        "assert ToolsConfig().web.search.__class__ is WebSearchConfig\n"
        "assert ToolsConfig().exec.__class__ is ExecToolConfig\n"
        "assert ToolsConfig().image_generation.__class__ is ImageGenerationToolConfig\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_unknown_attributes_still_raise() -> None:
    import nanobot.config.schema as schema

    with pytest.raises(AttributeError):
        schema.NotAConfig  # noqa: B018
