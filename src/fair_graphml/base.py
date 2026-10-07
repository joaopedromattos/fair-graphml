"""Small dependency-light implementation of the sklearn estimator protocol."""
from __future__ import annotations

import copy
import inspect

try:
    from sklearn.exceptions import NotFittedError
except ImportError:
    class NotFittedError(ValueError, AttributeError):
        """Raised when an estimator is used before fitting."""


class BaseEstimator:
    """Supports sklearn.clone, parameter search, and nested Pipeline parameters."""

    def get_params(self, deep=True):
        params = {}
        for cls in type(self).__mro__:
            init = cls.__dict__.get("__init__")
            if init is None:
                continue
            for name, p in inspect.signature(init).parameters.items():
                if name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                    params[name] = getattr(self, name)
        params.update(getattr(self, "method_params", {}))
        if deep:
            for name, value in list(params.items()):
                if hasattr(value, "get_params"):
                    params.update((f"{name}__{k}", v) for k, v in value.get_params().items())
        return params

    def set_params(self, **params):
        valid = self.get_params(deep=True)
        for name, value in params.items():
            key, _, subkey = name.partition("__")
            if key not in valid:
                raise ValueError(f"Unknown parameter {key!r} for {type(self).__name__}.")
            if subkey:
                valid[key].set_params(**{subkey: value})
            elif key in getattr(self, "method_params", {}):
                self.method_params[key] = value
            else:
                setattr(self, key, value)
        return self

    def __sklearn_clone__(self):
        return type(self)(**copy.deepcopy(self.get_params(deep=False)))

    def __sklearn_is_fitted__(self):
        return getattr(self, "is_fitted_", False)

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params(False).items())
        return f"{type(self).__name__}({args})"

    def _check_fitted(self):
        if not self.__sklearn_is_fitted__():
            raise NotFittedError(f"Call fit before using {type(self).__name__}.")


class TransformerMixin:
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)
