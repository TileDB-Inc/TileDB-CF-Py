"""Create a name registry for use in modifying grouped objects."""

from __future__ import annotations

from typing import Optional, TypeVar

from typing_extensions import Protocol, Self

T = TypeVar("T")


class Registry(Protocol[T]):
    def __delitem__(self, name: str):
        """Delete the element with the provided name."""

    def __getitem__(self, name: str) -> T:
        """Get the element with the provided name."""

    def __setitem__(self, name: str, value: T):
        """Set the elemetn with the provided name to the provided value."""

    def rename(self, old_name: str, new_name: str):
        """Rename an element of the registry.

        If the rename fails, the registry should be left unchanged.
        """


class RegisteredByNameMixin:
    def __init__(self, name: str, registry: Optional[Registry[Self]]):
        self._name = name
        self._registry: Optional[Registry[Self]] = None
        self.set_registry(registry)

    @property
    def is_registered(self) -> bool:
        return self._registry is not None

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        if self._registry is not None:
            self._registry.rename(self.name, name)
        self._name = name

    def set_registry(self, registry: Optional[Registry[Self]]):
        if self._registry is not None:
            raise ValueError("Registry is already set.")
        if registry is not None:
            registry[self.name] = self
        self._registry = registry
