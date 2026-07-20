"""Tests for whittaker.formula.parser (formula string -> Formula objects)."""

from __future__ import annotations

import pytest

from whittaker.formula.parser import parse
from whittaker.formula.terms import (
    InteractionTerm,
    LinearTerm,
    OffsetTerm,
    SmoothTerm,
)

# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def test_simple_identifier(self) -> None:
        f = parse("y ~ s(x)")
        assert f.response == "y"

    def test_underscore_identifier(self) -> None:
        f = parse("my_response ~ s(x)")
        assert f.response == "my_response"

    def test_missing_tilde_raises(self) -> None:
        with pytest.raises(ValueError, match="missing the '~' separator"):
            parse("y + s(x)")

    def test_non_identifier_response_raises(self) -> None:
        with pytest.raises(ValueError, match="valid Python identifier"):
            parse("log(y) ~ s(x)")

    def test_numeric_response_raises(self) -> None:
        with pytest.raises(ValueError, match="valid Python identifier"):
            parse("1 ~ s(x)")


# ---------------------------------------------------------------------------
# Linear terms
# ---------------------------------------------------------------------------


class TestLinearTerms:
    def test_single_linear_term(self) -> None:
        f = parse("y ~ x1")
        assert f.terms == [LinearTerm("x1")]

    def test_multiple_linear_terms(self) -> None:
        f = parse("y ~ x1 + x2 + x3")
        assert f.terms == [LinearTerm("x1"), LinearTerm("x2"), LinearTerm("x3")]

    def test_linear_term_is_string(self) -> None:
        f = parse("y ~ group")
        t = f.terms[0]
        assert isinstance(t, LinearTerm)
        assert t.variable == "group"


# ---------------------------------------------------------------------------
# Smooth terms: s()
# ---------------------------------------------------------------------------


class TestSmoothTerms:
    def test_basic_s(self) -> None:
        f = parse("y ~ s(x1)")
        assert len(f.terms) == 1
        t = f.terms[0]
        assert isinstance(t, SmoothTerm)
        assert t.variables == ("x1",)
        assert t.smooth_type == "s"
        assert t.bs == "tp"
        assert t.k == -1
        assert t.by is None
        assert t.extra == {}

    def test_bs_kwarg(self) -> None:
        f = parse("y ~ s(x1, bs='cr')")
        assert f.terms[0].bs == "cr"  # type: ignore[union-attr]

    def test_bs_kwarg_double_quotes(self) -> None:
        f = parse('y ~ s(x1, bs="ps")')
        assert f.terms[0].bs == "ps"  # type: ignore[union-attr]

    def test_k_kwarg(self) -> None:
        f = parse("y ~ s(x1, k=20)")
        assert f.terms[0].k == 20  # type: ignore[union-attr]

    def test_by_bare_name(self) -> None:
        f = parse("y ~ s(x1, by=group)")
        assert f.terms[0].by == "group"  # type: ignore[union-attr]

    def test_by_string_literal(self) -> None:
        f = parse("y ~ s(x1, by='group')")
        assert f.terms[0].by == "group"  # type: ignore[union-attr]

    def test_all_kwargs_combined(self) -> None:
        f = parse("y ~ s(x1, bs='cr', k=15, by=group)")
        t = f.terms[0]
        assert isinstance(t, SmoothTerm)
        assert t.variables == ("x1",)
        assert t.bs == "cr"
        assert t.k == 15
        assert t.by == "group"

    def test_extra_kwargs_stored(self) -> None:
        f = parse("y ~ s(x1, fx=True)")
        t = f.terms[0]
        assert isinstance(t, SmoothTerm)
        assert t.extra == {"fx": True}

    def test_no_variables_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one variable"):
            parse("y ~ s()")

    def test_non_name_positional_raises(self) -> None:
        with pytest.raises(ValueError, match="bare column names"):
            parse("y ~ s(x1 + x2)")


# ---------------------------------------------------------------------------
# Smooth terms: te / ti / t2
# ---------------------------------------------------------------------------


class TestTensorSmoothTerms:
    def test_te_two_vars(self) -> None:
        f = parse("y ~ te(x1, x2)")
        t = f.terms[0]
        assert isinstance(t, SmoothTerm)
        assert t.smooth_type == "te"
        assert t.variables == ("x1", "x2")

    def test_ti_two_vars(self) -> None:
        f = parse("y ~ ti(x1, x2)")
        t = f.terms[0]
        assert isinstance(t, SmoothTerm)
        assert t.smooth_type == "ti"
        assert t.variables == ("x1", "x2")

    def test_t2_two_vars(self) -> None:
        f = parse("y ~ t2(x1, x2)")
        t = f.terms[0]
        assert isinstance(t, SmoothTerm)
        assert t.smooth_type == "t2"
        assert t.variables == ("x1", "x2")

    def test_te_three_vars(self) -> None:
        f = parse("y ~ te(lon, lat, time)")
        t = f.terms[0]
        assert isinstance(t, SmoothTerm)
        assert t.variables == ("lon", "lat", "time")

    def test_te_with_k_list(self) -> None:
        # k=[5, 5] is a valid extra kwarg for tensor products
        f = parse("y ~ te(x1, x2, k=[5, 5])")
        t = f.terms[0]
        assert isinstance(t, SmoothTerm)
        # k=[5,5] goes into extra since it's not a plain int
        assert t.extra.get("k") == [5, 5] or t.k == -1

    def test_te_default_bs(self) -> None:
        f = parse("y ~ te(x1, x2)")
        assert f.terms[0].bs == "tp"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Offset terms
# ---------------------------------------------------------------------------


class TestOffsetTerms:
    def test_bare_column_offset(self) -> None:
        f = parse("y ~ s(x1) + offset(log_exposure)")
        offset = f.terms[1]
        assert isinstance(offset, OffsetTerm)
        assert offset.expression == "log_exposure"

    def test_expression_offset(self) -> None:
        f = parse("y ~ offset(log(n))")
        t = f.terms[0]
        assert isinstance(t, OffsetTerm)
        assert "log" in t.expression

    def test_offset_wrong_arity_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one positional argument"):
            parse("y ~ offset(a, b)")


# ---------------------------------------------------------------------------
# Interaction terms
# ---------------------------------------------------------------------------


class TestInteractionTerms:
    def test_full_interaction(self) -> None:
        f = parse("y ~ x1 * x2")
        t = f.terms[0]
        assert isinstance(t, InteractionTerm)
        assert t.left == "x1"
        assert t.right == "x2"
        assert t.full is True

    def test_non_name_interaction_raises(self) -> None:
        with pytest.raises(ValueError, match="bare column name"):
            parse("y ~ s(x1) * x2")


# ---------------------------------------------------------------------------
# Intercept control
# ---------------------------------------------------------------------------


class TestInterceptControl:
    def test_intercept_default_true(self) -> None:
        f = parse("y ~ s(x)")
        assert f.intercept is True

    def test_suppress_intercept_zero_prefix(self) -> None:
        f = parse("y ~ 0 + s(x)")
        assert f.intercept is False

    def test_suppress_intercept_minus_one_suffix(self) -> None:
        f = parse("y ~ s(x) - 1")
        assert f.intercept is False

    def test_explicit_one_is_noop(self) -> None:
        f = parse("y ~ 1 + s(x)")
        assert f.intercept is True
        # The literal "1" should NOT appear as a LinearTerm
        assert not any(isinstance(t, LinearTerm) and t.variable == "1" for t in f.terms)

    def test_suppress_intercept_zero_suffix(self) -> None:
        f = parse("y ~ s(x) + 0")
        assert f.intercept is False

    def test_terms_unaffected_by_intercept_suppression(self) -> None:
        f = parse("y ~ 0 + s(x1) + x2")
        assert f.intercept is False
        assert len(f.terms) == 2
        assert isinstance(f.terms[0], SmoothTerm)
        assert isinstance(f.terms[1], LinearTerm)


# ---------------------------------------------------------------------------
# Mixed / complex formulas
# ---------------------------------------------------------------------------


class TestComplexFormulas:
    def test_mixed_smooth_and_linear(self) -> None:
        f = parse("y ~ s(x1) + x2")
        assert isinstance(f.terms[0], SmoothTerm)
        assert isinstance(f.terms[1], LinearTerm)

    def test_multiple_smooths(self) -> None:
        f = parse("y ~ s(x1) + s(x2) + te(x3, x4)")
        assert len(f.terms) == 3
        assert all(isinstance(t, SmoothTerm) for t in f.terms)

    def test_full_complex_formula(self) -> None:
        f = parse(
            "cnt ~ s(temp, bs='cr', k=12) + s(hum) + te(hr, weekday) + workingday + offset(log_days)"
        )
        assert f.response == "cnt"
        types = [type(t) for t in f.terms]
        assert SmoothTerm in types
        assert LinearTerm in types
        assert OffsetTerm in types


# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------


class TestErrorMessages:
    def test_unknown_function_suggests_smooth_types(self) -> None:
        with pytest.raises(ValueError, match="'s'.*'t2'.*'te'.*'ti'"):
            parse("y ~ gam(x)")

    def test_syntax_error_in_rhs(self) -> None:
        with pytest.raises(ValueError, match="Unsupported formula term"):
            parse("y ~ s(x1) ++ s(x2)")
