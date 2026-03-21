"""
Forecaster plugin registry.

Models register themselves by name.  The pipeline selects which models to
run by reading ``forecast.forecasters`` from the platform config.

Usage
-----
Register a new forecaster::

    @registry.register("my_model")
    class MyForecaster(BaseForecaster):
        name = "my_model"
        ...

Instantiate from config::

    forecasters = registry.build_from_config(["naive_seasonal", "lgbm_direct"])
"""

import inspect
from typing import Callable, Dict, List, Optional, Type

from .base import BaseForecaster


class ForecasterRegistry:
    """Central registry of available forecasters."""

    def __init__(self):
        self._registry: Dict[str, Type[BaseForecaster]] = {}

    def register(
        self, name: str
    ) -> Callable[[Type[BaseForecaster]], Type[BaseForecaster]]:
        """
        Decorator to register a forecaster class.

        Usage::

            @registry.register("naive_seasonal")
            class SeasonalNaiveForecaster(BaseForecaster):
                ...
        """
        def decorator(cls: Type[BaseForecaster]) -> Type[BaseForecaster]:
            cls.name = name
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> Type[BaseForecaster]:
        if name not in self._registry:
            raise KeyError(
                f"Forecaster {name!r} not registered. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def build(self, name: str, **kwargs) -> BaseForecaster:
        """Instantiate a forecaster by name.

        Unknown kwargs are silently dropped so that pipeline-level
        parameters (e.g. ``frequency``) can be broadcast to all models
        without requiring every model to accept them.
        """
        cls = self.get(name)
        sig = inspect.signature(cls.__init__)
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if not accepts_var_kw:
            valid = set(sig.parameters) - {"self"}
            kwargs = {k: v for k, v in kwargs.items() if k in valid}
        return cls(**kwargs)

    def build_from_config(
        self,
        names: List[str],
        params: Optional[Dict[str, dict]] = None,
    ) -> List[BaseForecaster]:
        """
        Instantiate multiple forecasters from a list of names.

        Parameters
        ----------
        names:
            Forecaster names (must be registered).
        params:
            Optional per-forecaster kwargs. Keys are forecaster names.
        """
        params = params or {}
        forecasters = []
        for name in names:
            kwargs = params.get(name, {})
            forecasters.append(self.build(name, **kwargs))
        return forecasters

    @property
    def available(self) -> List[str]:
        return list(self._registry.keys())


# Global singleton
registry = ForecasterRegistry()
