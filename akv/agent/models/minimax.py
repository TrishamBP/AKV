"""MiniMax token counter with tracing matched to its chat template."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .jinja import JinjaTokenCounter

if TYPE_CHECKING:
    from jinja2 import nodes


class MiniMaxTokenCounter(JinjaTokenCounter):
    """Reuse Jinja rendering while preserving MiniMax's text-processing macro."""

    def __init__(self, model_path: str | Path) -> None:
        super().__init__(model_path, preserve_macro=preserve_format_macro)


VISIBLE_TEXT_TEMPLATE = """
{% macro visible_text(content) %}
{% if content is string %}{{ content }}
{% elif content is iterable and content is not mapping %}
{% for item in content %}
{% if item is mapping and item.type == 'text' %}{{ item.text }}
{% elif item is string %}{{ item }}{% endif %}
{% endfor %}
{% else %}{{ content }}{% endif %}
{% endmacro %}
"""


def preserve_format_macro(node: nodes.Macro) -> bool:
    """Keep MiniMax's pure text macro intact before the template splits its output."""
    if node.name != "visible_text":
        return False
    import copy

    from jinja2 import Environment, nodes
    from jinja2.visitor import NodeTransformer

    class StripWhitespace(NodeTransformer):
        def visit_TemplateData(
            self, item: nodes.TemplateData, *args: Any, **kwargs: Any
        ) -> nodes.TemplateData | None:
            return None if not item.data.strip() else item

        def visit_Output(self, item: nodes.Output, *args: Any, **kwargs: Any) -> nodes.Output | None:
            item = self.generic_visit(item, *args, **kwargs)
            return item if item.nodes else None

    expected = next(Environment().parse(VISIBLE_TEXT_TEMPLATE).find_all(nodes.Macro))
    if StripWhitespace().visit(copy.deepcopy(node)).dump() != StripWhitespace().visit(expected).dump():
        raise ValueError("Unrecognized MiniMax visible_text macro")
    return True
