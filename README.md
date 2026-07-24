# A/B Test Analyzer

A small web app for planning an experiment and then reading the result without fooling yourself. It works from summary numbers (conversions and sample sizes, or mean/sd/n), so you can check a test without pulling the raw event log.

> Live demo: [add link once deployed]

<!-- ![screenshot](docs/screenshot.png) -->

## What it does

- **Plan a test.** Given your current conversion rate, the smallest lift worth catching, your significance level and power, it tells you how many users each arm needs.
- **Read a conversion result.** Two-proportion z-test with a confidence interval on the lift, plus a Bayesian read (P(B beats A) and expected loss) so you get a probability you can actually act on, not just a p-value.
- **Read a continuous result.** Welch's t-test for metrics like revenue or time on page, with a confidence interval on the difference.
- **Catch broken tests first.** A sample-ratio-mismatch (SRM) check flags when users were not split the way you intended, which usually means a logging or assignment bug rather than a real effect.

## Why I built it

Most A/B test mistakes are not math errors, they are judgment errors: peeking at the p-value and stopping early, ignoring a broken split, or reading "not significant" as "no difference". I wanted one place that does the calculation and points out those traps, and building it kept my own statistics sharp.

## Run it

```bash
git clone https://github.com/divyantpratap/ab-test-analyzer.git
cd ab-test-analyzer
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py          # opens http://localhost:8501
```

## Tests

The statistics are checked against known values (hand-computed z-scores, symmetry when the two arms are identical, and sanity checks on sample size and SRM).

```bash
python -m pytest
```

## How the numbers are computed

- Two-proportion test: pooled standard error for the z-statistic, unpooled standard error for the confidence interval on the difference.
- Welch's t-test: unequal-variance t with Welch-Satterthwaite degrees of freedom.
- Sample size: normal-approximation formula using the variance at both the baseline and the target rate.
- Bayesian conversion: Beta-Binomial with a uniform prior, sampled from the posteriors to estimate P(B > A) and expected loss.
- SRM: chi-square goodness-of-fit on the two bucket counts against the intended split.

## License

MIT. See [LICENSE](LICENSE).
