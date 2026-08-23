"""Bootstrap intervals, kappa, and the gate decision."""

from __future__ import annotations

import math

import pytest

from prism.eval.stats import (
    bootstrap_ci,
    cohens_kappa,
    interpret_kappa,
    paired_bootstrap_ci,
    regression_detected,
    wilson_interval,
)


def test_bootstrap_interval_brackets_the_point_estimate():
    values = [0.0, 1.0] * 50
    ci = bootstrap_ci(values, resamples=2000)
    assert ci.low < ci.point < ci.high
    assert math.isclose(ci.point, 0.5)


def test_the_interval_is_deterministic_for_the_same_data():
    # A CI that moves between runs invites re-rolling until the answer is the
    # one you wanted — precisely what a quality gate exists to prevent.
    values = [0.0, 1.0, 1.0, 0.0, 1.0] * 20
    assert bootstrap_ci(values, resamples=2000) == bootstrap_ci(values, resamples=2000)


def test_small_samples_produce_wide_intervals():
    # The whole argument for reporting intervals: 20 examples cannot resolve a
    # small difference, and the interval is what says so.
    small = bootstrap_ci([1.0] * 12 + [0.0] * 8, resamples=2000)
    large = bootstrap_ci(([1.0] * 12 + [0.0] * 8) * 25, resamples=2000)
    assert small.width > large.width * 3


def test_a_three_percent_difference_on_two_hundred_examples_is_usually_noise():
    """The headline claim, pinned — and its real precondition.

    Whether +3% is noise depends on the *disagreement pattern*, not the margin.
    The realistic case is an arm that wins some examples and loses others,
    netting +3%; there the interval spans zero and the change has not been shown.
    """
    baseline = [1.0] * 140 + [0.0] * 60
    # Candidate wins 13 examples the baseline lost and loses 7 it won: net +3pp.
    candidate = list(baseline)
    for i in range(140, 153):
        candidate[i] = 1.0
    for i in range(7):
        candidate[i] = 0.0
    assert sum(candidate) - sum(baseline) == 6  # +3 percentage points

    diff = paired_bootstrap_ci(candidate, baseline, resamples=2000)
    assert diff.point == pytest.approx(0.03)
    assert not diff.excludes_zero
    assert not regression_detected(diff)


def test_a_uniform_three_percent_gain_with_no_losses_is_not_noise():
    """The flip side, and why pairing is worth the trouble.

    When the candidate wins 6 examples and loses none, the paired difference has
    almost no variance and +3% *is* detectable on 200 examples. Reporting "3% is
    always noise" would be as wrong as reporting the point estimate alone.
    """
    baseline = [1.0] * 140 + [0.0] * 60
    candidate = [1.0] * 146 + [0.0] * 54
    diff = paired_bootstrap_ci(candidate, baseline, resamples=2000)
    assert diff.excludes_zero


def test_pairing_is_tighter_than_treating_the_arms_independently():
    # Both arms saw the same examples, so an example hard for one is hard for
    # the other. Keeping the pairing removes that shared difficulty from the
    # variance; discarding it hides real effects behind example noise.
    baseline = [float(i % 7 == 0) for i in range(200)]
    candidate = [min(1.0, b + 0.05) for b in baseline]

    paired = paired_bootstrap_ci(candidate, baseline, resamples=2000)
    unpaired_spread = bootstrap_ci(candidate, resamples=2000).width + bootstrap_ci(
        baseline, resamples=2000
    ).width
    assert paired.width < unpaired_spread


def test_a_real_improvement_is_detected():
    baseline = [0.0] * 200
    candidate = [1.0] * 180 + [0.0] * 20
    diff = paired_bootstrap_ci(candidate, baseline, resamples=2000)
    assert diff.excludes_zero
    assert diff.low > 0


def test_the_gate_fires_only_when_the_whole_interval_is_a_regression():
    baseline = [1.0] * 190 + [0.0] * 10
    candidate = [1.0] * 100 + [0.0] * 100
    diff = paired_bootstrap_ci(candidate, baseline, resamples=2000)
    assert regression_detected(diff)

    # A gate that fired on the point estimate alone would fail roughly half of
    # all no-op changes and be switched off within a week.
    noise = paired_bootstrap_ci(
        [1.0] * 99 + [0.0] * 101, [1.0] * 100 + [0.0] * 100, resamples=2000
    )
    assert noise.point < 0
    assert not regression_detected(noise)


def test_tolerance_allows_a_deliberate_small_tradeoff():
    baseline = [1.0] * 160 + [0.0] * 40
    candidate = [1.0] * 150 + [0.0] * 50
    diff = paired_bootstrap_ci(candidate, baseline, resamples=2000)
    assert regression_detected(diff, tolerance=0.0)
    # e.g. a cheaper tier accepted at up to 10 points of quality loss.
    assert not regression_detected(diff, tolerance=0.10)


def test_bootstrap_handles_degenerate_samples():
    assert math.isnan(bootstrap_ci([]).point)
    single = bootstrap_ci([0.7])
    assert single.point == single.low == single.high == 0.7


def test_paired_comparison_rejects_mismatched_arms():
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap_ci([1.0, 0.0], [1.0])


# --- Cohen's kappa --------------------------------------------------------


def test_perfect_agreement_is_one():
    labels = ["A", "B", "tie", "A", "B"]
    assert cohens_kappa(labels, labels) == pytest.approx(1.0)


def test_skewed_agreement_is_corrected_toward_chance():
    # Two raters who both say "tie" most of the time agree often by accident.
    # Raw agreement here is 80%; kappa must be far lower.
    a = ["tie"] * 8 + ["A", "B"]
    b = ["tie"] * 8 + ["B", "A"]
    raw = sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)
    assert raw == pytest.approx(0.8)
    kappa = cohens_kappa(a, b)
    # 80% agreement reads as only "moderate" once chance is subtracted — which
    # is the entire reason raw agreement is not the number to report.
    assert kappa < raw - 0.35
    assert interpret_kappa(kappa) == "moderate"


def test_systematic_disagreement_is_negative():
    a = ["A", "A", "A", "B", "B", "B"]
    b = ["B", "B", "B", "A", "A", "A"]
    assert cohens_kappa(a, b) < 0


def test_kappa_is_undefined_when_everything_gets_one_label():
    # Total agreement, zero information. 1.0 would be a lie.
    assert math.isnan(cohens_kappa(["tie"] * 10, ["tie"] * 10))


def test_kappa_bands_are_reported_with_the_number():
    assert interpret_kappa(0.85) == "almost perfect"
    assert interpret_kappa(0.65) == "substantial"
    assert interpret_kappa(0.45) == "moderate"
    assert interpret_kappa(-0.1) == "worse than chance"
    assert interpret_kappa(float("nan")) == "undefined"


def test_kappa_requires_the_same_examples():
    with pytest.raises(ValueError, match="same examples"):
        cohens_kappa(["A", "B"], ["A"])


# --- Wilson interval ------------------------------------------------------


def test_zero_observed_failures_is_not_zero_percent():
    # A 0/20 false-hit rate does not mean the false-hit rate is zero, and the
    # semantic cache's operating point depends on knowing how wide that is.
    ci = wilson_interval(0, 20)
    assert ci.point == 0.0
    assert ci.high > 0.10


def test_the_interval_tightens_as_the_sample_grows():
    assert wilson_interval(5, 20).width > wilson_interval(250, 1000).width


def test_agreement_at_one_hundred_labels_carries_real_uncertainty():
    # ~100 labels is the usual human budget; the interval is roughly +-10 points.
    ci = wilson_interval(78, 100)
    assert ci.low < 0.72 and ci.high > 0.84
