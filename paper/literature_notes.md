# Literature Notes: Minimum Wage Effects and Staggered Difference-in-Differences

**Project:** Auto-Econ-Research — Minimum Wage Employment Effects
**Date Compiled:** 2026-03-15
**Author:** Auto-generated research synthesis

---

## Overview

This document synthesizes the key literature on two interrelated topics: (1) the employment effects of minimum wage increases, and (2) methodological advances in staggered difference-in-differences (DiD) estimation that are essential for credible causal inference in minimum wage research. The papers are organized thematically, moving from foundational empirical debates to modern estimation methods and recent post-COVID evidence.

---

## 1. Staggered Difference-in-Differences Methodology

### 1.1 Callaway & Sant'Anna (2021)

**Title:** Difference-in-Differences with Multiple Time Periods
**Authors:** Brantly Callaway, Pedro H. C. Sant'Anna
**Journal:** Journal of Econometrics
**Volume/Issue/Pages:** Vol. 225, No. 2, pp. 200–230
**Year:** 2021
**DOI:** 10.1016/j.jeconom.2020.12.001

**Key Finding:** Callaway and Sant'Anna demonstrate that conventional two-way fixed effects (TWFE) DiD estimators produce biased and potentially sign-reversing estimates when treatment timing is staggered and treatment effects are heterogeneous. They introduce *group-time average treatment effects* (ATT(g,t)) as the building block for identification, with clean "not-yet-treated" or "never-treated" comparison groups. They provide estimation via outcome regression, inverse probability weighting, and doubly-robust methods, together with aggregation schemes (e.g., event-study plots, average effects by treatment cohort) that recover economically meaningful causal parameters.

**Relevance:** This paper is the primary methodological framework for estimating minimum wage effects in staggered policy settings. It resolves the "negative weights" problem that plagues TWFE in the presence of heterogeneous treatment effects and has become a standard tool in the empirical labor economics toolkit.

**BibTeX key:** `callaway2021difference`

---

### 1.2 Sun & Abraham (2021)

**Title:** Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects
**Authors:** Liyang Sun, Sarah Abraham
**Journal:** Journal of Econometrics
**Volume/Issue/Pages:** Vol. 225, No. 2, pp. 175–199
**Year:** 2021
**DOI:** 10.1016/j.jeconom.2020.09.006

**Key Finding:** Sun and Abraham show that in a staggered adoption setting, the standard TWFE event-study regression produces coefficients on leads and lags that are contaminated by effects from *other* periods and *other* cohorts, due to negative weighting of treatment effects from already-treated units used as controls. Apparent pre-trends in TWFE estimates can be spurious artifacts of treatment effect heterogeneity rather than evidence of parallel-trends violations. They propose a fully interacted estimator (the "interaction-weighted" estimator) that cleanly identifies cohort-specific dynamic effects without contamination.

**Relevance:** Directly addresses the validity of pre-trend tests and dynamic treatment effect estimation in minimum wage event studies. The proposed estimator is implemented in the `eventstudyinteract` Stata package and is widely adopted in applied work.

**BibTeX key:** `sun2021estimating`

---

## 2. Classic and Foundational Minimum Wage Research

### 2.1 Card & Krueger (1994)

**Title:** Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania
**Authors:** David Card, Alan B. Krueger
**Journal:** American Economic Review
**Volume/Issue/Pages:** Vol. 84, No. 4, pp. 772–793
**Year:** 1994

**Key Finding:** Using telephone surveys of 410 fast-food restaurants before and after New Jersey raised its minimum wage from \$4.25 to \$5.05 in April 1992 — while neighboring Pennsylvania kept wages unchanged — Card and Krueger find no evidence that the minimum wage increase reduced employment. Fast-food employment actually grew slightly more in New Jersey relative to Pennsylvania, contradicting the standard competitive model prediction of disemployment.

**Relevance:** A landmark natural experiment that re-opened the empirical debate on minimum wages. Introduced the contiguous-state quasi-experimental design that became standard in the field. Its results have been revisited, replicated, and challenged extensively, but the paper's methodology continues to influence research design.

**BibTeX key:** `card1994minimum`

---

### 2.2 Neumark & Wascher (2000)

**Title:** Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania: Comment
**Authors:** David Neumark, William Wascher
**Journal:** American Economic Review
**Volume/Issue/Pages:** Vol. 90, No. 5, pp. 1362–1396
**Year:** 2000
**DOI:** 10.1257/aer.90.5.1362

**Key Finding:** Neumark and Wascher re-examine the Card-Krueger (1994) New Jersey experiment using administrative payroll data from BurgerKing, KFC, Wendy's, and Roy Rogers franchises rather than telephone surveys. They find significant *negative* employment effects from the minimum wage increase, arguing that Card and Krueger's telephone-based data contained substantial measurement error that biased results toward zero. This paper is part of a broader literature by Neumark and Wascher documenting negative employment effects of minimum wages, especially for low-skilled, young, and part-time workers.

**Relevance:** Represents the principal empirical counterpoint to Card-Krueger and encapsulates the "disemployment hypothesis" position in the minimum wage debate. Highlights the critical role of data quality and measurement in causal inference.

**BibTeX key:** `neumark2000minimum`

---

### 2.3 Dube, Lester & Reich (2010)

**Title:** Minimum Wage Effects Across State Borders: Estimates Using Contiguous Counties
**Authors:** Arindrajit Dube, T. William Lester, Michael Reich
**Journal:** Review of Economics and Statistics
**Volume/Issue/Pages:** Vol. 92, No. 4, pp. 945–964
**Year:** 2010
**DOI:** 10.1162/REST_a_00039

**Key Finding:** Dube, Lester, and Reich construct a novel control group by comparing employment outcomes in counties on opposite sides of state borders — which share local economic conditions but face different minimum wage regimes. Using all contiguous county-pair combinations in the U.S. from 1990 to 2006, they find *no* employment effect of minimum wage increases on restaurant and retail employment, and no evidence of reduced hours. They show that the negative employment estimates in earlier studies are artifacts of comparison to distant counties experiencing different economic trends, rather than true causal effects.

**Relevance:** A methodological breakthrough that addresses the endogeneity of minimum wage timing to state-level economic conditions. The contiguous-county approach is now standard in the field. This paper shifted the consensus toward smaller or zero employment effects for moderate minimum wage increases.

**BibTeX key:** `dube2010minimum`

---

## 3. Wage Distribution and Low-Wage Employment

### 3.1 Cengiz, Dube, Lindner & Zipperer (2019)

**Title:** The Effect of Minimum Wages on Low-Wage Jobs
**Authors:** Doruk Cengiz, Arindrajit Dube, Attila Lindner, Ben Zipperer
**Journal:** Quarterly Journal of Economics
**Volume/Issue/Pages:** Vol. 134, No. 3, pp. 1405–1454
**Year:** 2019
**DOI:** 10.1093/qje/qjz014

**Key Finding:** Using a novel bunching estimator that examines excess mass in the wage distribution just above minimum wage thresholds, the authors study 138 state-level minimum wage changes from 1979 to 2016. They find that minimum wage increases substantially raised wages for affected workers while the overall number of low-wage jobs remained essentially unchanged over the five years following each increase. The study rules out labor-labor substitution (e.g., adults replacing teenagers) as a mechanism, and finds reduced employment only in tradable sectors — consistent with a competitive model of internationally exposed industries.

**Relevance:** Methodologically innovative in using the wage distribution as a direct measure of minimum wage "bite." Provides the most comprehensive and credible evidence to date that employment effects are near zero, and is widely cited as a key reference in policy debates.

**BibTeX key:** `cengiz2019effect`

---

### 3.2 Autor, Manning & Smith (2016)

**Title:** The Contribution of the Minimum Wage to US Wage Inequality over Three Decades: A Reassessment
**Authors:** David H. Autor, Alan Manning, Christopher L. Smith
**Journal:** American Economic Journal: Applied Economics
**Volume/Issue/Pages:** Vol. 8, No. 1, pp. 58–99
**Year:** 2016
**DOI:** 10.1257/app.20140073

**Key Finding:** Reassessing the role of minimum wages in U.S. wage inequality using additional decades of CPS data and an IV strategy to address bias from using lagged wages as instruments, Autor et al. find that minimum wages do reduce lower-tail wage inequality. However, their estimates are substantially *smaller* than those of DiNardo, Fortin, and Lemieux (1996), suggesting that rising lower-tail inequality since 1980 is primarily driven by underlying structural wage changes (skill-biased technical change, declining unionization) rather than minimum wage erosion. They also detect spillover effects above the statutory minimum but caution these may partly reflect measurement artifacts.

**Relevance:** Provides a quantitative decomposition of minimum wage contributions to U.S. wage inequality, situating minimum wage policy within the broader distributional trends of the past four decades.

**BibTeX key:** `autor2016contribution`

---

## 4. Recent Evidence: Machine Learning Methods and Post-COVID Labor Markets (2021–2024)

### 4.1 Cengiz, Dube, Lindner & Zentler-Munro (2022)

**Title:** Seeing Beyond the Trees: Using Machine Learning to Estimate the Impact of Minimum Wages on Labor Market Outcomes
**Authors:** Doruk Cengiz, Arindrajit Dube, Attila S. Lindner, David Zentler-Munro
**Journal:** Journal of Labor Economics
**Volume/Issue/Pages:** Vol. 40, No. S1, pp. S203–S247
**Year:** 2022
**DOI:** 10.1086/718497

**Key Finding:** This paper extends the Cengiz et al. (2019) bunching approach by using machine learning to identify a broader treatment group covering approximately 75% of minimum wage workers — compared to the narrow demographic proxies (e.g., teenagers) used in earlier studies. Analyzing 172 minimum wage increases between 1979 and 2019, they find a clear increase in average wages for affected workers and little evidence of employment loss. The results hold across demographic subgroups, including teenagers, older workers, and single mothers, and show no adverse effects on labor force participation or job search behavior.

**Relevance:** An important methodological update that addresses the treatment group definition problem in minimum wage research. The machine learning approach reduces misclassification of affected workers and provides more representative estimates of minimum wage effects for the policy-relevant population.

**BibTeX key:** `cengiz2022seeing`

---

### 4.2 Autor, Dube & McGrew (2023)

**Title:** The Unexpected Compression: Competition at Work in the Low Wage Labor Market
**Authors:** David Autor, Arindrajit Dube, Annie McGrew
**Institution:** National Bureau of Economic Research Working Paper No. 31010
**Year:** 2023 (revised 2024)
**DOI:** 10.3386/w31010

**Key Finding:** This paper documents a striking structural change in U.S. wage inequality following the COVID-19 pandemic. Post-pandemic labor market tightness drove rapid relative wage growth at the bottom of the distribution, reducing the college wage premium and offsetting approximately one-third of four decades of aggregate 90/10 wage inequality growth. Wage gains were concentrated among young non-college workers who switched employers, consistent with a mechanism of reduced employer monopsony power — specifically, an increase in the elasticity of labor supply to firms in the low-wage labor market. The authors emphasize that these gains reflected market forces (labor scarcity) rather than minimum wage policy per se.

**Relevance:** Provides essential context for interpreting minimum wage research in the post-COVID period: the unusual tightness of the low-wage labor market after 2020 means that the counterfactual for minimum wage effects differs markedly from pre-pandemic conditions. Understanding the interaction of minimum wage policy with tight labor markets is a priority for future research.

**BibTeX key:** `autor2023unexpected`

---

## 5. Summary Table

| BibTeX Key | Authors | Year | Journal | Main Contribution |
|---|---|---|---|---|
| `callaway2021difference` | Callaway & Sant'Anna | 2021 | J. Econometrics | Group-time ATTs for staggered DiD |
| `sun2021estimating` | Sun & Abraham | 2021 | J. Econometrics | Interaction-weighted estimator for event studies |
| `card1994minimum` | Card & Krueger | 1994 | AER | NJ-PA natural experiment; no disemployment |
| `neumark2000minimum` | Neumark & Wascher | 2000 | AER | Payroll data shows negative employment effects |
| `dube2010minimum` | Dube, Lester & Reich | 2010 | REStat | Contiguous-county design; no employment effect |
| `cengiz2019effect` | Cengiz et al. | 2019 | QJE | Bunching estimator; near-zero employment effects |
| `autor2016contribution` | Autor, Manning & Smith | 2016 | AEJ: Applied | Minimum wage reduces lower-tail inequality |
| `cengiz2022seeing` | Cengiz et al. | 2022 | J. Labor Econ. | ML treatment groups; confirms near-zero effects |
| `autor2023unexpected` | Autor, Dube & McGrew | 2023 | NBER WP | Post-COVID wage compression in low-wage markets |

---

## 6. Key Methodological Themes

### 6.1 The Parallel Trends Assumption and Control Group Selection

The central challenge in minimum wage research is constructing a credible counterfactual: what would have happened to employment in treated states/counties in the absence of the minimum wage increase? Early studies used national trends as the comparison, but Dube et al. (2010) demonstrate that minimum wage increases are systematically timed to state economic cycles, making national comparisons confounded. The contiguous-county and bunching estimator approaches directly address this by using geographic proximity or the wage distribution structure to achieve comparability.

### 6.2 Staggered Treatment Timing and Heterogeneous Effects

With states adopting minimum wages at different times and different levels, staggered DiD designs are unavoidable. Callaway and Sant'Anna (2021) and Sun and Abraham (2021) — appearing in the same issue of the Journal of Econometrics — provide complementary solutions to the TWFE contamination problem. In practice, minimum wage event studies should use one of these methods to avoid mischaracterizing pre-trends or treatment effect dynamics.

### 6.3 Treatment Group Definition

A persistent challenge is identifying *which* workers are affected by a given minimum wage increase. Using narrow proxies (teenagers, restaurant workers) understates the affected population and reduces statistical power. Cengiz et al. (2022) demonstrate that machine learning methods can identify a broader, more representative treatment group covering approximately 75% of minimum wage workers, yielding more externally valid estimates.

### 6.4 Post-COVID Labor Market Conditions

Autor, Dube, and McGrew (2023) document that post-pandemic conditions fundamentally altered the low-wage labor market, compressing the wage distribution through market forces. Future minimum wage research must account for this structural shift: the pre-trend assumption and the appropriate comparison group may differ markedly in this environment. Studies published in 2021–2024 that do not explicitly address the post-COVID labor market context may produce estimates that are not representative of the current policy environment.

---

## 7. Research Gaps and Future Directions

1. **Post-COVID minimum wage effects:** There is limited high-quality causal evidence on how minimum wage increases interact with the unusually tight post-pandemic labor market. Studies using 2021–2024 data and credible staggered DiD designs are needed.

2. **Long-run effects:** Most studies focus on 1–5 year horizons. Evidence on long-run (10+ year) employment, wage, and human capital effects is scarce.

3. **Geographic heterogeneity:** Optimal minimum wage levels likely vary by local labor market conditions. More work is needed on heterogeneous effects by local unemployment rate, cost of living, and industry composition.

4. **Firm-level adjustment margins:** How firms absorb minimum wage increases through prices, profits, hours, benefits, and automation remains an active area. Harasztosi and Lindner (2019, AER) provide evidence from Hungary showing most costs pass through to consumers.

5. **Monopsony and employer market power:** The Autor, Dube, and McGrew (2023) findings on post-COVID wage compression suggest that employer monopsony power in low-wage markets is more significant than standard competitive models imply, with important implications for the employment effects of minimum wages.

---

*End of literature notes. See `references.bib` for full BibTeX citations.*
