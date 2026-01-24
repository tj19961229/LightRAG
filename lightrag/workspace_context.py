"""
Workspace context management for dynamic multi-tenant support.

This module provides a context-variable based mechanism for dynamically
switching workspaces without modifying storage class method signatures.

Usage:
    from lightrag.workspace_context import workspace_context, set_workspace

    # In LightRAG main class methods:
    async def aquery(self, query, param=None, workspace=None):
        with set_workspace(workspace or self.workspace):
            # All storage operations will use the context workspace
            ...

    # In storage classes:
    @property
    def effective_workspace(self):
        return get_effective_workspace(self.workspace)
"""

from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from typing import Optional

# Context variable for current workspace
# This is async-safe and provides isolation between concurrent requests
_workspace_context: ContextVar[Optional[str]] = ContextVar(
    "workspace_context", default=None
)


def get_workspace_context() -> Optional[str]:
    """Get the current workspace from context.

    Returns:
        The workspace set in current context, or None if not set.
    """
    return _workspace_context.get()


def get_effective_workspace(default_workspace: str) -> str:
    """Get the effective workspace, preferring context over default.

    Args:
        default_workspace: The default workspace from storage instance.

    Returns:
        Context workspace if set, otherwise the default workspace.
    """
    context_workspace = _workspace_context.get()
    return context_workspace if context_workspace is not None else default_workspace


@contextmanager
def set_workspace(workspace: str):
    """Context manager for temporarily setting the workspace.

    This is async-safe and can be used with concurrent requests.
    Each coroutine will have its own workspace context.

    Args:
        workspace: The workspace to use within this context.

    Yields:
        None

    Example:
        with set_workspace("user_123"):
            await rag.aquery("question")  # Uses workspace "user_123"
    """
    token = _workspace_context.set(workspace)
    try:
        yield
    finally:
        _workspace_context.reset(token)


async def async_set_workspace(workspace: str):
    """Async context manager for temporarily setting the workspace.

    Same as set_workspace but as an async context manager.

    Args:
        workspace: The workspace to use within this context.
    """
    return set_workspace(workspace)
