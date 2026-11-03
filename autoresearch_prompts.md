# Autoresearch prompts: seeds from the best-performing fails

Ready-to-run prompts for autoresearch agents, distilled from the five
strongest failed experiments, aimed at the two frontier tips (#171 on the
gnn-stack tree, #174 on the gaussian-MLP tree).

## 1. RCD on the gnn tree (from #163's finding)

> let's work on rcd on top of this work
> https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/171
> with the contribution model use this framework notes/autoresearch.md the
> direction pointed by
> https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/163
> is that rcd is bought by free-running exposure but scheduled sampling is
> vetoed on the iteration budget so find a budget-legal route to the same
> exposure be creative don't get too hinged to ideas we need diversity in
> things we try also staying grounded in criteria set in the framework

## 2. CG residual on the gmlp tree via a trained group latent (from #159)

> let's work on cg on top of this work
> https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/174
> with the contribution model use this framework notes/autoresearch.md get
> inspiration from
> https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/159
> the trained episode-persistent group latent was the best cg fail on record
> and only its sampling-time cousin was ever adopted mind the lesson of
> https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/166
> that latents do not compose without joint recalibration but be creative
> don't get too hinged to ideas we need diversity in things we try also
> staying grounded in criteria set in the framework

## 3. Richer emission under the group copula on the gmlp tree (from #154)

> let's work on the c marginals and cc on top of this work
> https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/174
> with the contribution model use this framework notes/autoresearch.md get
> inspiration from
> https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/154
> whose xgboost emission had the best likelihood ever and band-upgraded cg
> and rcd but lacked trajectory memory — both halves now exist separately on
> this tree (nonlinear emission #167, marginal-preserving copula #170) be
> creative don't get too hinged to ideas we need diversity in things we try
> also staying grounded in criteria set in the framework

## 4. CE and the singleton residual on the gmlp tree (from #173/#175)

> let's work on ce and the singleton exodus residual on top of this work
> https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/174
> with the contribution model use this framework notes/autoresearch.md the
> behavioural finding of
> https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/173
> (singletons polarise, merged eights sag) is solid but the plain feature
> encoding failed twice (#173, #175) and #174 showed one-hot size codes beat
> scalars so condition the emission on size structurally be creative don't
> get too hinged to ideas we need diversity in things we try also staying
> grounded in criteria set in the framework

## 5. Calibrated conformity dose on the gnn tree (from #157)

> let's work on cg on top of this work
> https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/171
> with the contribution model use this framework notes/autoresearch.md get
> inspiration from
> https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/157
> whose conformity mixture was the best cg move of its era but teacher
> forcing pinned the gate at w~0.08 — set the dose by calibration against
> the human data the way every copula rho was estimated, not by mle, and
> mind that schedsamp successors are vetoed be creative don't get too hinged
> to ideas we need diversity in things we try also staying grounded in
> criteria set in the framework

## 6. CG via multi-aggregator message passing on the gnn tree (from GIN/PNA)

> let's work on cg on top of this work
> https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/171
> with the contribution model use this framework notes/autoresearch.md the
> trunk is a 2018-style graph net whose scatter_mean aggregation provably
> cannot carry a neighborhood's spread or extremes (xu et al. 2019 gin,
> corso et al. 2020 pna) which is exactly what cg measures — replace the
> mean with a pna-style multi-aggregator (mean max min std) at the
> aggregation sites and recalibrate the copula on the retrained trunk per
> the #166 lesson, mind #153 which showed attention alone re-weights the
> mean without fixing this be creative don't get too hinged to ideas we
> need diversity in things we try also staying grounded in criteria set in
> the framework

## 7. CG via a recurrent per-group virtual node on the gnn tree (from MPNN)

> let's work on cg on top of this work
> https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/171
> with the contribution model use this framework notes/autoresearch.md add
> a learned per-group virtual node (gilmer et al. 2017 master node, the
> standard ogb trick) that aggregates members and broadcasts group state
> back each round, with its own gru so the group state persists across the
> episode — the architectural cousin of the three best cg mechanisms so
> far (#167's group feature core, #159's episode-persistent group latent,
> #170's group copula); the existing GlobalModel is global across all
> agents, the literature move is per-group; recalibrate the copula on the
> retrained trunk per the #166 lesson be creative don't get too hinged to
> ideas we need diversity in things we try also staying grounded in
> criteria set in the framework
