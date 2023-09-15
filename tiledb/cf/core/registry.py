from typing import Optional, Protocol, Self, TypeVar

T = TypeVar("T")


class Registry(Protocol[T]):
    def __delitem__(self, name: str):
        ...

    def __getitem__(self, name: str) -> T:
        ...

    def __setitem__(self, name: str, value: T):
        ...

    def rename(self, old_name: str, new_name: str):
        """Rename an element of the registry.

        If the rename fails, the registry should be left unchanged.
        """
        ...


class RegisteredByName:
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
