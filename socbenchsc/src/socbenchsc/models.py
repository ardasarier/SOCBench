from __future__ import annotations
from pydantic import BaseModel
from typing import Callable
from pydantic.dataclasses import dataclass


class FunctionCall(BaseModel):
    name: str
    arguments: list[Variable] = []
    kwargs: dict[str, Variable] = {}


@dataclass(frozen=True)
class Variable:
    values: frozenset[Unknown | Constant | Function | List | Dict | RequestImport]

    def __str__(self):
        if len(self.values) == 1:
            return "'" + str(list(self.values)[0]) + "'"
        return str(set(self.values))

    def __repr__(self):
        return self.__str__()


@dataclass(frozen=True)
class Unknown:
    def __hash__(self):
        return hash(self.__class__.__name__)

    def __str__(self):
        return ""

    def __repr__(self):
        return ""


@dataclass(frozen=True)
class Constant:
    value: str

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"'{self.value}'"


@dataclass(frozen=True)
class List:
    values: tuple[Variable, ...]

    def __str__(self):
        return str(list(self.values))

    def __repr__(self):
        return str(list(self.values))


@dataclass(frozen=True)
class Dict:
    value: frozenset[tuple[Variable, Variable]] = frozenset()

    def lookup(self, lookup_key: Variable) -> Variable:
        result = set()
        for current_key in lookup_key.values:
            for key, value in self.value:
                if current_key in key.values:
                    result.update(value.values)
        if len(result) > 0:
            return Variable(values=frozenset(result))
        return UNKNOWN

    def add(self, in_key: Variable, in_item: Variable) -> Dict:
        result = list(self.value)
        for value in in_key.values:
            found = False
            for key, item in result:
                if value in key.values:
                    result.remove((key, item))
                    item = Variable(values=item.values | in_item.values)
                    result.append((key, item))
                    found = True
                    break
            if not found:
                result.append((Variable(values=frozenset({value})), in_item))
        return Dict(value=frozenset(result))

    def remove(self, key: Variable) -> Dict:
        result = list(self.value)
        for value in key.values:
            for key, item in result:
                if value in key.values:
                    result.remove((key, item))
        return Dict(value=tuple(result))


class Function(BaseModel):
    transform: Callable | None = None
    variables: dict[str, Variable] = {}
    parent: Function | None = None
    returns: Variable | None = None

    def lookup_value(self, variable) -> Variable:
        if variable in self.variables:
            return self.variables[variable]
        if self.parent:
            return self.parent.lookup_value(variable)
        return UNKNOWN

    def __hash__(self):
        return hash(self.transform)


@dataclass(frozen=True)
class RequestImport:
    pass


UNKNOWN = Variable(values=frozenset({Unknown()}))


class Invocation(BaseModel):
    """One service invocation found in the code: the endpoint it calls and
    every parameter it passes, as computed by the reaching definition
    analysis. perform_analysis() keeps only method and path; this keeps the
    rest, e.g. to check the call against the endpoint's schema."""
    method: str
    path: str
    arguments: dict[str, Variable] = {}

    def keys(self, name: str) -> tuple[frozenset[str], bool]:
        """Keys of the dictionary passed as parameter `name` (params, json,
        data, headers, ...), and whether they are completely known.

        A parameter that is not passed has no keys and is complete. The keys
        of all possible values are united, so the result overapproximates,
        like the endpoint analysis itself. Any possible value that is not a
        dictionary with constant keys -- Unknown, a string body, a computed
        key -- makes the result incomplete."""
        if name not in self.arguments:
            return frozenset(), True
        keys, complete = set(), True
        for value in self.arguments[name].values:
            if isinstance(value, Dict):
                for key, _ in value.value:
                    for key_value in key.values:
                        if isinstance(key_value, Constant):
                            keys.add(key_value.value)
                        else:
                            complete = False
            else:
                complete = False
        return frozenset(keys), complete
