# SSRN submission pack

Everything the form asks for, in the order it asks. Copy and paste.

---

## 1. Paper title

```
When Does Limit Order Book Information Constitute Tradable Alpha?
```

## 2. Subtitle (if the form offers one)

```
Explicit predictive gates, cost-aware evaluation, cross-sectional inference, and reproducibility
```

## 3. Date written

```
August 2026
```

## 4. Abstract

Paste as plain text. SSRN strips formatting, so this version has no markdown.

```
Limit order book signals are routinely reported as "predictive" on the strength of a rank correlation and a rising equity curve. That claim is usually underspecified: it is rarely clear whether the effect holds on each instrument or only on the most active one, whether the significance survives the number of hypotheses actually tested, or whether the prediction is large enough to pay the spread it must cross. This paper develops a protocol for making those distinctions explicit.

Using five NASDAQ instruments from the LOBSTER sample (AAPL, AMZN, GOOG, INTC, and MSFT; 953,829 clean events), I evaluate four microstructure signals against four sequential gates: cross-sectional predictive power, survival of multiple-testing correction, profitability under a parameterised cost model, and superiority over a block-shifted null.

Three of the four signals clear the first three gates decisively, with rank information coefficients between 0.10 and 0.19 that hold the same sign on every instrument. None is profitable. The best signal breaks even only if execution costs 4.8% of the assumed half-spread, and costs consume 2,099% of gross profit under the baseline assumption.

I also find that queue imbalance and micro-price deviation are the same signal in different clothing (rank correlation 0.969, incremental out-of-sample R squared of -0.0006), and that order flow imbalance and the top-of-book signals dominate on opposite ends of the liquidity spectrum: order flow on wide-spread, thin-queue names, and queue imbalance on penny-spread, deep-queue names, with the ordering reversing completely between the two groups.

A complete Python implementation, with 80 tests and synthetic controls of known ground truth, is provided.

AI disclosure: Anthropic's Claude was used extensively in producing this work, including implementing the accompanying Python framework, running the analysis pipeline, and drafting and editing the manuscript. The research question, project design, signal selection, data acquisition, and the decision of what to report were directed by the author, who verified every reported figure against the pipeline's output tables.
```

## 5. Keywords

```
limit order book, market microstructure, order flow imbalance, queue imbalance, micro-price, transaction costs, high-frequency trading, multiple testing, backtest overfitting, reproducible research
```

## 6. JEL codes

Optional on SSRN and it does not affect distribution, but it makes the paper browsable.

| Code | Area |
|:---|:---|
| **G14** | Information and market efficiency, event studies |
| **G17** | Financial forecasting and simulation |
| **C58** | Financial econometrics |
| **C52** | Model evaluation, validation, and selection |
| **C12** | Hypothesis testing |

## 7. eJournals

Pick from the Financial Economics Network list at submission time. Names shift, so match on topic rather than trusting these strings exactly. In rough priority order:

1. Capital Markets: Market Microstructure
2. Econometric Modeling: Capital Markets, Asset Pricing
3. Econometrics: Mathematical Methods and Programming
4. Machine Learning eJournal, if the framework framing is emphasised

Choose deliberately rather than leaving it to SSRN. Papers surface slowly, or not at all, in the right topic areas when the selection is left automatic.

## 8. Author details

```
William Odumosu
Independent Researcher
<your email>
```

SSRN requires an affiliation and will not accept a blank one. "Independent Researcher" is a standard, accepted value, and it is what the paper's title page says, so the two must match.

---

## Before you upload

- [ ] Replace the email placeholder on the title page of the PDF, and use the same address on the SSRN form
- [ ] Confirm `github.com/study-holic/q1-lob-alpha` is public and pushed before the paper posts
- [ ] Push the repository public first, so the link resolves on day one
- [ ] Read the AI disclosure and confirm it is accurate (see below)
- [ ] Confirm the PDF opens and the title page shows your name and affiliation

## On the AI disclosure

SSRN's guidelines require an AI disclosure statement when AI is used, and specify that it must appear with the abstract and on the PDF itself. Both are done.

Read the wording carefully before submitting. It is deliberately unflattering, because an understated disclosure is the kind of thing that damages a research reputation permanently and a candid one costs almost nothing. If anything in it overstates or understates what actually happened, change it. What matters is that it is true.

## What happens next

SSRN reviews submissions before posting, typically within one to three days. Once approved you get a permanent abstract page and a stable link, and downloads start accruing.

Post the same PDF to arXiv q-fin.TR in parallel. Two records of the same preprint is standard, and arXiv is where the microstructure audience actually is.
