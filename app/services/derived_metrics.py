"""Derived measurements: formulas a dataset declares over its direct measurements.

The registry metrics are *direct*: code measures them off an outline. A derived metric is
arithmetic over those, plus aggregation over related objects -- which is what most
domain indices turn out to be:

* choroidal vascularity index = sum(area of the choroid's Vessel children) / area
* polyp density               = count(the coral's Polyp children) / area

A formula is a small, checked tree (:data:`Expression`), never free text, so nothing here
evaluates user input as code. Five kinds of node:

``metric``    a measurement of the object itself (a registry key or another derived key)
``constant``  a number
``rollup``    an aggregate (sum / mean / min / max / count) of a measurement over the
              object's ``children``, ``descendants``, ``parent`` or every object on the same
              ``image``, optionally only those of some labels
``binary``    ``+ - * /`` of two nodes
``func``      ``sqrt``, ``abs``, ``log``, ``log10`` of one node

Units follow from the formula. Every value carries a **length exponent** -- 1 for a
length, 2 for an area, 0 for a count or ratio -- and the arithmetic carries it along:
dividing a count by an area gives -2, i.e. "per mm²". That is also how nonsense is caught
before anyone sees a number (an area plus a length has no unit at all).

Evaluation runs on values already converted to the display unit (see
``app.services.database_access.datasets``), so a derived number is in exactly the unit its
inputs are reported in and follows the same all-or-nothing scale rule. Related objects
always share an image -- children and descendants live on their parent's mask -- so no
formula ever mixes two images' scales.

A label in a formula means exactly that label. The label tree is not a taxonomy here --
Vessel sits under Choroid because vessels lie *inside* choroids -- so a label never stands
for the labels below it; a formula that wants several picks several.

This module is pure: it knows the spec, checks it and evaluates it on pandas data. Fetching
the data is the database layer's job.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Annotated, Iterable, Literal, Union

import numpy as np
import pandas as pd
from iquana_toolbox.quantification import METRIC_REGISTRY
from iquana_toolbox.quantification.registry import UnitKind
from pydantic import BaseModel, Field

#: Kept small on purpose: a formula is meant to be read on one line of the editor.
MAX_NODES = 64
MAX_DEPTH = 12

Scope = Literal["children", "descendants", "parent", "image"]
Aggregation = Literal["sum", "mean", "min", "max", "count"]


class MetricOperand(BaseModel):
    """The object's own value of a direct or derived measurement."""
    type: Literal["metric"] = "metric"
    metric: str = Field(..., min_length=1, max_length=64)
    component: int = Field(default=0, ge=0, description="For multi-component metrics, e.g. L of Lab.")


class ConstantOperand(BaseModel):
    """A plain number. Unitless."""
    type: Literal["constant"] = "constant"
    value: float = Field(..., allow_inf_nan=False)


class RollupOperand(BaseModel):
    """An aggregate of a measurement over objects related to this one."""
    type: Literal["rollup"] = "rollup"
    scope: Scope
    agg: Aggregation
    label_ids: list[int] | None = Field(
        default=None, description="Only related objects of these labels; null = any.")
    metric: str | None = Field(default=None, max_length=64, description="Not used by count.")
    component: int = Field(default=0, ge=0)


class BinaryOperation(BaseModel):
    type: Literal["binary"] = "binary"
    op: Literal["+", "-", "*", "/"]
    left: "Expression"
    right: "Expression"


class FunctionCall(BaseModel):
    type: Literal["func"] = "func"
    fn: Literal["sqrt", "abs", "log", "log10"]
    arg: "Expression"


Expression = Annotated[
    Union[MetricOperand, ConstantOperand, RollupOperand, BinaryOperation, FunctionCall],
    Field(discriminator="type"),
]
BinaryOperation.model_rebuild()
FunctionCall.model_rebuild()


class DerivedMetricDefinition(BaseModel):
    """A derived measurement as a dataset declares it."""
    key: str = Field(..., pattern=r"^[a-z][a-z0-9_]{0,47}$",
                     description="Column name in exports and the name other formulas use.")
    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(default="", max_length=1024)
    label_ids: list[int] | None = Field(
        default=None, description="Labels the measurement is reported on; null = all.")
    expression: Expression


class DerivedMetricError(ValueError):
    """A definition that cannot be used, with every reason found."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


# --- Names ------------------------------------------------------------------------

#: Columns of the flat export that are not metrics. A derived key may not shadow them.
_FIXED_COLUMNS = frozenset({
    "file_name", "label", "label_id", "contour_id", "parent_id", "parent_label",
    "coords_x", "coords_y", "diameter_avg",
})
_RESERVED_PREFIXES = ("meta_", "attr_", "rel_")


def reserved_names() -> set[str]:
    """Every name a derived key would collide with in an export or a formula."""
    names = set(_FIXED_COLUMNS)
    for key, metric in METRIC_REGISTRY.items():
        names.add(key)
        for index in range(metric.value_dim):
            names.add(f"{key}_{index}")
            if metric.components and index < len(metric.components):
                names.add(f"{key}_{metric.components[index].lower()}")
    return names


def check_key_available(key: str, existing_keys: Iterable[str]) -> list[str]:
    """Problems with using ``key`` for a new derived metric (empty when it is free)."""
    if key in reserved_names() or key.startswith(_RESERVED_PREFIXES):
        return [f"'{key}' is already the name of a built-in measurement or export column."]
    if key in set(existing_keys):
        return [f"This dataset already has a derived measurement called '{key}'."]
    return []


# --- Units --------------------------------------------------------------------------

def metric_exponent(metric_key: str) -> float:
    """Length exponent of a registry metric: 1 for lengths, 2 for areas, else 0."""
    kind = METRIC_REGISTRY[metric_key].unit_kind
    if kind == UnitKind.LENGTH:
        return 1.0
    if kind == UnitKind.AREA:
        return 2.0
    return 0.0


_SUPERSCRIPTS = {2: "²", 3: "³"}


def unit_for_exponent(exponent: float, display_unit: str) -> str:
    """Unit string for a value of length exponent ``exponent``: ``mm``, ``mm²``, ``/mm²``.

    Fractional exponents (a square root of a length) are written out, ``mm^0.5``.
    """
    if abs(exponent) < 1e-9:
        return ""
    magnitude = abs(exponent)
    if abs(magnitude - round(magnitude)) < 1e-9:
        power = int(round(magnitude))
        body = display_unit + (_SUPERSCRIPTS.get(power, f"^{power}") if power != 1 else "")
    else:
        body = f"{display_unit}^{magnitude:g}"
    return body if exponent > 0 else f"/{body}"


# --- Checking -----------------------------------------------------------------------

def _walk(node, depth: int = 1):
    """Yield ``(node, depth)`` for every node of an expression tree."""
    yield node, depth
    if isinstance(node, BinaryOperation):
        yield from _walk(node.left, depth + 1)
        yield from _walk(node.right, depth + 1)
    elif isinstance(node, FunctionCall):
        yield from _walk(node.arg, depth + 1)


def referenced_keys(expression) -> set[str]:
    """Every measurement key a formula reads, direct or derived."""
    keys = set()
    for node, _ in _walk(expression):
        if isinstance(node, MetricOperand):
            keys.add(node.metric)
        elif isinstance(node, RollupOperand) and node.agg != "count" and node.metric:
            keys.add(node.metric)
    return keys


def referenced_label_ids(definition: DerivedMetricDefinition) -> set[int]:
    """Every label a definition names, in its scope or inside its formula."""
    ids = set(definition.label_ids or [])
    for node, _ in _walk(definition.expression):
        if isinstance(node, RollupOperand) and node.label_ids:
            ids.update(node.label_ids)
    return ids


def _structural_problems(definition: DerivedMetricDefinition, known_derived: set[str],
                         dataset_label_ids: set[int]) -> list[str]:
    """What is wrong with a definition on its own, before units and dependencies."""
    problems: list[str] = []
    nodes = list(_walk(definition.expression))
    if len(nodes) > MAX_NODES:
        problems.append(f"The formula is too long ({len(nodes)} parts; at most {MAX_NODES}).")
    if max(depth for _, depth in nodes) > MAX_DEPTH:
        problems.append(f"The formula is nested too deeply (at most {MAX_DEPTH} levels).")

    missing_labels = referenced_label_ids(definition) - dataset_label_ids
    if missing_labels:
        problems.append("It refers to labels that no longer exist in this dataset "
                        f"(ids {', '.join(str(i) for i in sorted(missing_labels))}).")
    if definition.label_ids is not None and not definition.label_ids:
        problems.append("Pick at least one label to report it on, or report it on all labels.")

    for node, _ in nodes:
        if isinstance(node, RollupOperand):
            if node.agg != "count" and not node.metric:
                problems.append(f"A {node.agg} over {node.scope} needs a measurement to aggregate.")
                continue
            if node.label_ids is not None and not node.label_ids:
                problems.append(f"A rollup over {node.scope} filters by labels but none are picked.")
        if not isinstance(node, (MetricOperand, RollupOperand)):
            continue
        if isinstance(node, RollupOperand) and node.agg == "count":
            continue
        key = node.metric
        if key == definition.key:
            problems.append(f"'{key}' cannot refer to itself.")
        elif key in METRIC_REGISTRY:
            dim = METRIC_REGISTRY[key].value_dim
            if node.component >= dim:
                problems.append(f"'{key}' has {dim} component(s); component {node.component} does not exist.")
        elif key in known_derived:
            if node.component != 0:
                problems.append(f"Derived measurement '{key}' has a single value; it has no component "
                                f"{node.component}.")
        else:
            problems.append(f"Unknown measurement '{key}'.")
    return problems


def _exponent(node, derived_exponents: dict[str, float]) -> float:
    """Length exponent of a node, raising :class:`DerivedMetricError` on a unit clash."""
    if isinstance(node, ConstantOperand):
        return 0.0
    if isinstance(node, (MetricOperand, RollupOperand)):
        if isinstance(node, RollupOperand) and node.agg == "count":
            return 0.0
        if node.metric in METRIC_REGISTRY:
            return metric_exponent(node.metric)
        return derived_exponents[node.metric]
    if isinstance(node, FunctionCall):
        inner = _exponent(node.arg, derived_exponents)
        if node.fn == "sqrt":
            return inner / 2
        if node.fn == "abs":
            return inner
        if abs(inner) > 1e-9:
            raise DerivedMetricError([f"{node.fn}() needs a unitless value (a ratio or a count); "
                                      "the logarithm of a length or an area changes with the unit."])
        return 0.0
    left = _exponent(node.left, derived_exponents)
    right = _exponent(node.right, derived_exponents)
    if node.op in "+-":
        if abs(left - right) > 1e-9:
            raise DerivedMetricError([f"Cannot {'add' if node.op == '+' else 'subtract'} values of "
                                      f"different units ({_describe_exponent(left)} and "
                                      f"{_describe_exponent(right)})."])
        return left
    return left + right if node.op == "*" else left - right


def _describe_exponent(exponent: float) -> str:
    return {0.0: "unitless", 1.0: "a length", 2.0: "an area"}.get(exponent, unit_for_exponent(exponent, "px"))


@dataclass
class ResolvedDerivedMetric:
    """A stored definition together with what checking it found."""
    definition: DerivedMetricDefinition
    id: int | None = None
    exponent: float | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def resolve_definitions(
        definitions: list[tuple[int | None, DerivedMetricDefinition]],
        dataset_label_ids: set[int],
) -> tuple[list[ResolvedDerivedMetric], list[ResolvedDerivedMetric]]:
    """Check a dataset's definitions as a set.

    A definition can refer to another, so they are checked together: in dependency order,
    with a cycle or a broken dependency marking every definition that relies on it. A
    definition that refers to a deleted label keeps its row and reports the problem rather
    than disappearing, so the person who made it can see what happened and fix it.

    :returns: ``(all, evaluation_order)`` -- every definition in the given order with its
        problems and unit, and the usable ones in an order where each comes after
        everything it reads.
    """
    resolved = {d.key: ResolvedDerivedMetric(definition=d, id=i) for i, d in definitions}
    known = set(resolved)
    for item in resolved.values():
        item.problems.extend(_structural_problems(item.definition, known, dataset_label_ids))

    # A self-reference is reported by the structural check; leaving it out here keeps it
    # from being reported a second time as a one-element cycle.
    deps = {key: (referenced_keys(item.definition.expression) & known) - {key}
            for key, item in resolved.items()}
    order: list[str] = []
    state: dict[str, str] = {}  # "visiting" | "done"

    def visit(key: str, path: list[str]) -> None:
        if state.get(key) == "done":
            return
        if state.get(key) == "visiting":
            cycle = path[path.index(key):] + [key]
            for member in cycle[:-1]:
                message = f"Circular reference: {' → '.join(cycle)}."
                if message not in resolved[member].problems:
                    resolved[member].problems.append(message)
            return
        state[key] = "visiting"
        for dep in sorted(deps[key]):
            visit(dep, path + [key])
        state[key] = "done"
        order.append(key)

    for key in resolved:
        visit(key, [])

    exponents: dict[str, float] = {}
    for key in order:
        item = resolved[key]
        broken = [dep for dep in sorted(deps[key]) if not resolved[dep].ok]
        if broken and item.ok:
            item.problems.append(f"It uses {', '.join(repr(b) for b in broken)}, which "
                                 f"{'has' if len(broken) == 1 else 'have'} a problem.")
        if not item.ok:
            continue
        try:
            item.exponent = _exponent(item.definition.expression, exponents)
            exponents[key] = item.exponent
        except DerivedMetricError as exc:
            item.problems.extend(exc.problems)

    everything = [resolved[d.key] for _, d in definitions]
    evaluation_order = [resolved[key] for key in order if resolved[key].ok]
    return everything, evaluation_order


def direct_metric_inputs(definitions: Iterable[DerivedMetricDefinition]) -> set[tuple[str, int]]:
    """``(registry key, component)`` pairs the formulas read, so the caller fetches them."""
    pairs = set()
    for definition in definitions:
        for node, _ in _walk(definition.expression):
            if isinstance(node, RollupOperand) and node.agg == "count":
                continue
            if isinstance(node, (MetricOperand, RollupOperand)) and node.metric in METRIC_REGISTRY:
                pairs.add((node.metric, node.component))
    return pairs


# --- Evaluation ---------------------------------------------------------------------

@dataclass
class EvaluationFrame:
    """Everything a formula can read, for one set of objects.

    ``objects`` is indexed by contour id with ``label_id``, ``parent_id`` and ``mask_id``
    columns: exactly the objects being quantified (the page's filters already applied), and
    only those can contribute to a rollup. ``contour_parents`` is the hierarchy of the whole
    dataset, so a descendant still reaches its grandparent through a parent the filters
    left out. ``values`` maps ``(metric key, component)`` to a series over ``objects``.
    """
    objects: pd.DataFrame
    contour_parents: dict[int, int | None]
    values: dict[tuple[str, int], pd.Series]

    def members(self, label_ids: list[int] | None) -> pd.Index:
        if label_ids is None:
            return self.objects.index
        return self.objects.index[self.objects["label_id"].isin(label_ids)]


_EMPTY_IS_ZERO = {"sum", "count"}


def _aggregate(values: pd.Series, owners, agg: str) -> pd.Series:
    """Aggregate ``values`` by the owner each one belongs to."""
    if values.empty:
        return pd.Series(dtype=float)
    grouped = pd.Series(values.to_numpy(dtype=float), index=pd.Index(owners)).groupby(level=0)
    return {"sum": grouped.sum, "mean": grouped.mean, "min": grouped.min,
            "max": grouped.max, "count": grouped.sum}[agg]()


def _ancestors(contour_id: int, parents: dict[int, int | None], memo: dict[int, list[int]]) -> list[int]:
    """Every ancestor of a contour, nearest first. Guarded against a cyclic hierarchy."""
    if contour_id in memo:
        return memo[contour_id]
    chain, seen, current = [], {contour_id}, parents.get(contour_id)
    while current is not None and current not in seen:
        chain.append(current)
        if current in memo:
            chain.extend(memo[current])
            break
        seen.add(current)
        current = parents.get(current)
    memo[contour_id] = chain
    return chain


def _evaluate_rollup(node: RollupOperand, frame: EvaluationFrame, computed: dict[str, pd.Series]) -> pd.Series:
    index = frame.objects.index
    members = frame.members(node.label_ids)
    if node.agg == "count":
        member_values = pd.Series(1.0, index=members)
    else:
        member_values = _operand_values(node.metric, node.component, frame, computed).reindex(members)

    if node.scope == "parent":
        parent_ids = frame.objects["parent_id"]
        has_member_parent = parent_ids.isin(members).to_numpy()
        if node.agg == "count":
            return pd.Series(has_member_parent.astype(float), index=index)
        looked_up = member_values.reindex(parent_ids.to_numpy()).to_numpy(dtype=float)
        return pd.Series(np.where(has_member_parent, looked_up, np.nan), index=index)

    if node.scope == "children":
        owners = frame.objects.loc[members, "parent_id"].to_numpy()
        keep = pd.notna(owners)
        result = _aggregate(member_values[keep], owners[keep].astype(int), node.agg)
    elif node.scope == "descendants":
        memo: dict[int, list[int]] = {}
        owners, sources = [], []
        for member in members:
            for ancestor in _ancestors(int(member), frame.contour_parents, memo):
                owners.append(ancestor)
                sources.append(member)
        result = _aggregate(member_values.reindex(sources), owners, node.agg)
    else:  # image
        result = _aggregate(member_values, frame.objects.loc[members, "mask_id"].to_numpy(), node.agg)
        result = pd.Series(result.reindex(frame.objects["mask_id"].to_numpy()).to_numpy(), index=index)
        return result.fillna(0.0) if node.agg in _EMPTY_IS_ZERO else result

    result = result.reindex(index)
    return result.fillna(0.0) if node.agg in _EMPTY_IS_ZERO else result


def _operand_values(key: str, component: int, frame: EvaluationFrame,
                    computed: dict[str, pd.Series]) -> pd.Series:
    if key in computed:
        return computed[key]
    series = frame.values.get((key, component))
    if series is None:
        return pd.Series(np.nan, index=frame.objects.index)
    return series.reindex(frame.objects.index).astype(float)


def _evaluate(node, frame: EvaluationFrame, computed: dict[str, pd.Series]) -> pd.Series:
    index = frame.objects.index
    if isinstance(node, ConstantOperand):
        return pd.Series(float(node.value), index=index)
    if isinstance(node, MetricOperand):
        return _operand_values(node.metric, node.component, frame, computed)
    if isinstance(node, RollupOperand):
        return _evaluate_rollup(node, frame, computed)
    if isinstance(node, FunctionCall):
        arg = _evaluate(node.arg, frame, computed).to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = {"sqrt": lambda a: np.sqrt(np.where(a >= 0, a, np.nan)),
                   "abs": np.abs,
                   "log": lambda a: np.log(np.where(a > 0, a, np.nan)),
                   "log10": lambda a: np.log10(np.where(a > 0, a, np.nan))}[node.fn](arg)
        return pd.Series(out, index=index)
    left = _evaluate(node.left, frame, computed).to_numpy(dtype=float)
    right = _evaluate(node.right, frame, computed).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        if node.op == "+":
            out = left + right
        elif node.op == "-":
            out = left - right
        elif node.op == "*":
            out = left * right
        else:
            out = np.where(right != 0, left / np.where(right != 0, right, 1.0), np.nan)
    return pd.Series(np.where(np.isfinite(out), out, np.nan), index=index)


def evaluate(evaluation_order: list[ResolvedDerivedMetric], frame: EvaluationFrame) -> dict[str, pd.Series]:
    """Compute every usable derived metric for every object of ``frame``.

    :returns: ``{key: series over frame.objects}``. An object outside a metric's labels, or
        whose value cannot be computed (a division by zero, a missing input), holds NaN.
    """
    computed: dict[str, pd.Series] = {}
    for item in evaluation_order:
        definition = item.definition
        values = _evaluate(definition.expression, frame, computed)
        if definition.label_ids is not None:
            in_scope = frame.objects.index.isin(frame.members(definition.label_ids))
            values = values.where(in_scope)
        computed[definition.key] = values.astype(float)
    return computed


def summarize(values: pd.Series) -> dict[str, float | int] | None:
    """count / mean / population std / min / max of the defined values (None when empty).

    The same statistics, with the same population std, as the SQL summary computes for the
    direct metrics, so a derived card reads like every other one.
    """
    finite = values.dropna().to_numpy(dtype=float)
    if finite.size == 0:
        return None
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def finite_or_none(value: float) -> float | None:
    """A JSON-safe value: NaN becomes None."""
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)
