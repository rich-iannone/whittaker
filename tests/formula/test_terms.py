"""Tests for whittaker.formula.terms (data classes and Formula helpers)."""

from __future__ import annotations

from whittaker.formula.terms import (
    Formula,
    InteractionTerm,
    LinearTerm,
    OffsetTerm,
    SmoothTerm,
)


class TestLinearTermRepr:
    def test_repr(self) -> None:
        assert repr(LinearTerm("age")) == "age"


class TestInteractionTermRepr:
    def test_full_interaction(self) -> None:
        t = InteractionTerm(left="x1", right="x2", full=True)
        assert repr(t) == "x1 * x2"

    def test_pure_interaction(self) -> None:
        t = InteractionTerm(left="x1", right="x2", full=False)
        assert repr(t) == "x1 : x2"


class TestOffsetTermRepr:
    def test_repr(self) -> None:
        assert repr(OffsetTerm("log_n")) == "offset(log_n)"


class TestSmoothTermRepr:
    def test_default_bs_omitted(self) -> None:
        t = SmoothTerm(variables=("x",))
        assert repr(t) == "s(x)"

    def test_non_default_bs_shown(self) -> None:
        t = SmoothTerm(variables=("x",), bs="cr")
        assert "bs='cr'" in repr(t)

    def test_k_shown(self) -> None:
        t = SmoothTerm(variables=("x",), k=20)
        assert "k=20" in repr(t)

    def test_by_shown(self) -> None:
        t = SmoothTerm(variables=("x",), by="group")
        assert "by='group'" in repr(t)

    def test_te_type(self) -> None:
        t = SmoothTerm(variables=("x1", "x2"), smooth_type="te")
        assert repr(t).startswith("te(")

    def test_extra_kwargs(self) -> None:
        t = SmoothTerm(variables=("x",), extra={"fx": True})
        assert "fx=True" in repr(t)


class TestFormulaRequiredColumns:
    def test_response_included(self) -> None:
        f = Formula(response="y", terms=[LinearTerm("x")])
        assert "y" in f.required_columns()

    def test_linear_term_included(self) -> None:
        f = Formula(response="y", terms=[LinearTerm("x1"), LinearTerm("x2")])
        assert f.required_columns() == ["y", "x1", "x2"]

    def test_smooth_variables_included(self) -> None:
        f = Formula(
            response="y",
            terms=[SmoothTerm(variables=("x1", "x2"), smooth_type="te")],
        )
        cols = f.required_columns()
        assert "x1" in cols
        assert "x2" in cols

    def test_by_variable_included(self) -> None:
        f = Formula(
            response="y",
            terms=[SmoothTerm(variables=("x",), by="group")],
        )
        assert "group" in f.required_columns()

    def test_interaction_term_included(self) -> None:
        f = Formula(
            response="y",
            terms=[InteractionTerm(left="a", right="b")],
        )
        cols = f.required_columns()
        assert "a" in cols
        assert "b" in cols

    def test_no_duplicates(self) -> None:
        f = Formula(
            response="y",
            terms=[
                SmoothTerm(variables=("x",)),
                SmoothTerm(variables=("x",), by="g"),
            ],
        )
        cols = f.required_columns()
        assert cols.count("x") == 1

    def test_offset_expression_skipped(self) -> None:
        f = Formula(response="y", terms=[OffsetTerm("log(n)")])
        # The expression "log(n)" is not a bare column name; it should not raise
        cols = f.required_columns()
        assert "y" in cols

    def test_order_preserved(self) -> None:
        f = Formula(
            response="y",
            terms=[LinearTerm("b"), LinearTerm("a")],
        )
        cols = f.required_columns()
        assert cols == ["y", "b", "a"]


class TestFormulaRepr:
    def test_with_intercept(self) -> None:
        f = Formula(response="y", terms=[LinearTerm("x")])
        assert repr(f) == "y ~ x"

    def test_without_intercept(self) -> None:
        f = Formula(response="y", terms=[LinearTerm("x")], intercept=False)
        assert repr(f).startswith("y ~ 0 +")

    def test_multiple_terms(self) -> None:
        f = Formula(
            response="y",
            terms=[SmoothTerm(variables=("x1",)), LinearTerm("x2")],
        )
        r = repr(f)
        assert "s(x1)" in r
        assert "x2" in r
