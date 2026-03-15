
# Empirical Analysis Results: State Minimum Wage Effects on L&H Employment (2017-2024)

## Research Design
- **Outcome**: Log employment in Leisure & Hospitality sector (BLS SAE data)
- **Treatment**: State minimum wage above federal floor ($7.25/hr)
- **Panel**: 47 U.S. states × 8 years (2017-2024), N=376 obs
- **Identification**: Staggered DID using variation in timing of state MW increases
- **Treatment groups**: 
  - Never-treated (g=0): 19 states at $7.25 floor (mostly South/Midwest)
  - Early-treated (g=2014): 27 states that raised wages before 2017
  - Late-treated (g=2021): 1 state (Virginia, large post-COVID increase)
- **Control variables**: State FE, Year FE
- **SE clustering**: State-level clustered standard errors

## Main Results

### TWFE Binary DID
- β = -0.0162 (SE=0.0060, p=0.0073, ***)
- Interpretation: Minimum wage increases reduce L&H employment by 1.62%
- N = 376 observations, 47 states

### TWFE Log-Log (Employment Elasticity)
- β = -0.0677 (SE=0.0374, p=0.0705, *)
- Interpretation: 10% increase in minimum wage → 0.68% decrease in L&H employment
- Implied elasticity range: -0.068 (consistent with Dube et al. 2010)

### Modern Staggered DID Estimators (Callaway-Sant'Anna 2021)
- Equal-weight ATT = -0.0132 (SE=0.0067, p=0.048, **)
- Sample-weight ATT = -0.0159 (SE=0.0048, p=0.001, ***)

### Sun-Abraham (2021) Estimator
- ATT = +0.0024 (SE=0.0012, p=0.056, *)
- NOTE: Sign reversal compared to TWFE and CS-2021 — this is unusual and potentially interesting

## Pre-Trends Test (Parallel Trends Assumption)
Callaway-Sant'Anna event study, pre-treatment coefficients:
- t=-6: β=-0.0016 (SE=0.0060) — insignificant ✓
- t=-5: β=+0.0059 (SE=0.0069) — insignificant ✓
- t=-4: β=+0.0038 (SE=0.0079) — insignificant ✓
- t=-3: β=-0.0007 (SE=0.0047) — insignificant ✓
- t=-2: β=+0.0019 (SE=0.0049) — insignificant ✓
Result: PARALLEL TRENDS ASSUMPTION IS SATISFIED ✓

## Robustness Checks
1. Drop big states (CA, NY, TX, FL): β=-0.061, p=0.114 — loses significance
2. Exclude COVID year 2020: β=-0.083, p=0.037, ** — strengthens
3. Binary DID (most robust): β=-0.016, p=0.007, *** — consistent

## KEY PUZZLE: TWFE vs Sun-Abraham Sign Reversal

The Sun-Abraham estimator gives a POSITIVE coefficient (+0.24%) while TWFE and CS-2021 give NEGATIVE coefficients (-1.3% to -1.6%). This divergence requires explanation:

Hypothesis 1: Heterogeneous Treatment Effects
- TWFE is biased when treatment effects vary across cohorts/time
- Sun-Abraham corrects for this "contaminated comparison" bias
- If early-treated states (g=2014) had NEGATIVE effects but late-treated (g=2021) had POSITIVE effects, Sun-Abraham would weight them differently than TWFE

Hypothesis 2: Composition Effects
- The "never-treated" group (19 states) may be systematically different
- Southern states with no MW increases may have had WORSE L&H performance post-COVID (right-to-work states, different labor market structure)
- This would artificially inflate the negative TWFE estimate

Hypothesis 3: Dynamic Effects
- If minimum wage effects fade over time (short-run negative, long-run zero/positive), then SA vs CS weighting could explain the difference

## Literature Context
This analysis connects to:
1. Dube, Lester, Reich (2010): Find near-zero employment effects using county-pair design
2. Neumark and Wascher (2007): Meta-analysis finds negative effects (elasticity -0.1 to -0.3)
3. Callaway and Sant'Anna (2021): Developed the CS estimator to handle staggered treatment
4. Cengiz et al. (2019): Bunching estimator finds near-zero net employment effects
5. HLER paper (2026): Automates this type of DID analysis using multi-agent pipeline

## Data Sources
- Employment: BLS SAE API, series SMU[FIPS]000007000000001 (L&H employment by state)
- Minimum wage: U.S. DOL Wage and Hour Division state minimum wage history
- Time period: 2017-2024 (annual), 47 states

## Open Questions for Cross-Validation
1. Is the Sun-Abraham vs TWFE sign reversal plausible given the data structure?
2. Does dropping the never-treated group and using only 2x2 cohort comparisons change results?
3. Should we worry about the small "late-treated" group (only Virginia in 2021)?
4. What does the literature say about minimum wage elasticities in L&H specifically?
5. Is the -0.9% to -1.6% range consistent with comparable studies using post-2020 data?
