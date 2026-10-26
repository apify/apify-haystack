from __future__ import annotations

import pytest
from haystack.core.serialization import generate_qualified_class_name, import_class_by_name

import apify_haystack

PUBLIC_CLASSES = [name for name in apify_haystack.__all__ if isinstance(getattr(apify_haystack, name), type)]


@pytest.mark.parametrize('name', [pytest.param(name, id=name) for name in PUBLIC_CLASSES])
def test_class_serializes_by_public_path(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """A public class serializes as `apify_haystack.<name>`, and Haystack imports it back from that path."""
    monkeypatch.setenv('HAYSTACK_DESERIALIZATION_ALLOWLIST', 'apify_haystack')
    cls = getattr(apify_haystack, name)

    assert generate_qualified_class_name(cls) == f'apify_haystack.{name}'
    assert import_class_by_name(f'apify_haystack.{name}') is cls
