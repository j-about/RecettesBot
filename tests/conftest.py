"""Root-level test configuration.

Fixtures that are truly shared across all test packages (unit, integration,
e2e) belong here. Note: the ``_base_env`` autouse fixture that sets fake
database URLs lives in ``tests/unit/conftest.py`` because integration tests
need real database URLs from ``.env``.
"""
