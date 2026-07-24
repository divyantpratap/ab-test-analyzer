"""A/B Test Analyzer, a small Streamlit app for planning and reading experiments."""
from __future__ import annotations

import streamlit as st

from abtest import (
    bayesian_conversion,
    relative_mde_to_absolute,
    sample_size_proportions,
    srm_check,
    two_proportion_test,
    welch_t_test,
)

st.set_page_config(page_title="A/B Test Analyzer", page_icon="AB", layout="centered")

st.title("A/B Test Analyzer")
st.caption(
    "Plan an experiment, then read the result without fooling yourself. "
    "Works from summary numbers, so you do not need the raw event log."
)

plan, read = st.tabs(["Plan a test", "Read a result"])


with plan:
    st.subheader("How many users do I need?")
    st.write(
        "Pick your current conversion rate and the smallest lift worth detecting. "
        "The calculator returns the users needed in each arm."
    )
    c1, c2 = st.columns(2)
    with c1:
        baseline = st.number_input("Current conversion rate (%)", 0.1, 99.0, 5.0, 0.1) / 100
        rel_mde = st.number_input("Smallest lift worth detecting (%, relative)", 0.5, 100.0, 10.0, 0.5) / 100
    with c2:
        alpha = st.selectbox("Significance level", [0.01, 0.05, 0.10], index=1)
        power = st.selectbox("Power", [0.8, 0.9, 0.95], index=0)

    abs_mde = relative_mde_to_absolute(baseline, rel_mde)
    try:
        per_arm = sample_size_proportions(baseline, abs_mde, alpha=alpha, power=power)
        st.metric("Users needed per arm", f"{per_arm:,}")
        st.metric("Total users", f"{per_arm * 2:,}")
        st.info(
            f"Detecting a move from {baseline * 100:.2f}% to {(baseline + abs_mde) * 100:.2f}% "
            f"at {int(power * 100)}% power. Halving the detectable lift roughly quadruples the users needed."
        )
    except ValueError as e:
        st.error(str(e))


with read:
    st.subheader("What does my result say?")
    metric_type = st.radio("Metric type", ["Conversion (yes/no)", "Continuous (revenue, time, etc.)"], horizontal=True)
    alpha_r = st.selectbox("Significance level ", [0.01, 0.05, 0.10], index=1, key="alpha_read")

    if metric_type.startswith("Conversion"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Control (A)**")
            n_a = st.number_input("Users A", 1, 10_000_000, 10_000, key="na")
            conv_a = st.number_input("Conversions A", 0, 10_000_000, 500, key="ca")
        with c2:
            st.markdown("**Variant (B)**")
            n_b = st.number_input("Users B", 1, 10_000_000, 10_000, key="nb")
            conv_b = st.number_input("Conversions B", 0, 10_000_000, 560, key="cb")

        if st.button("Analyze", type="primary"):
            if conv_a > n_a or conv_b > n_b:
                st.error("Conversions cannot exceed users.")
            else:
                srm = srm_check(int(n_a), int(n_b))
                (st.success if not srm.mismatch else st.error)(srm.summary())

                r = two_proportion_test(int(conv_a), int(n_a), int(conv_b), int(n_b), alpha=alpha_r)
                st.markdown("**Frequentist**")
                st.write(r.summary())
                verdict = "Significant" if r.significant else "Not significant"
                st.metric("Verdict", verdict, f"{r.rel_lift * 100:+.2f}% relative")

                b = bayesian_conversion(int(conv_a), int(n_a), int(conv_b), int(n_b))
                st.markdown("**Bayesian**")
                st.write(b.summary())
                st.progress(min(max(b.prob_b_beats_a, 0.0), 1.0), text=f"P(B > A) = {b.prob_b_beats_a * 100:.1f}%")

                if not r.significant:
                    st.warning(
                        "Not significant does not mean 'no difference', it means you do not have "
                        "enough evidence yet. Avoid peeking at the p-value daily and stopping the "
                        "moment it dips below your threshold, that inflates false positives."
                    )
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Control (A)**")
            mean_a = st.number_input("Mean A", value=10.0, key="ma")
            sd_a = st.number_input("Std dev A", 0.0, value=4.0, key="sa")
            na2 = st.number_input("n A", 2, 10_000_000, 1000, key="na2")
        with c2:
            st.markdown("**Variant (B)**")
            mean_b = st.number_input("Mean B", value=10.4, key="mb")
            sd_b = st.number_input("Std dev B", 0.0, value=4.0, key="sb")
            nb2 = st.number_input("n B", 2, 10_000_000, 1000, key="nb2")

        if st.button("Analyze", type="primary"):
            r = welch_t_test(mean_a, sd_a, int(na2), mean_b, sd_b, int(nb2), alpha=alpha_r)
            verdict = "Significant" if r.significant else "Not significant"
            st.metric("Verdict", verdict, f"{r.diff:+.3f} absolute")
            st.write(
                f"Welch t = {r.t:.3f}, df = {r.df:.0f}, p = {r.p_value:.4f}. "
                f"95% CI on the difference [{r.ci_low:.3f}, {r.ci_high:.3f}]."
            )

st.divider()
st.caption("Built by Divyant Pratap. Code: github.com/USERNAME/ab-test-analyzer")
