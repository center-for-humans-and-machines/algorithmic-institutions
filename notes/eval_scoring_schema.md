# The Evaluation Scoring Schema

How the raw evaluation metrics (#128, #134) become comparable scores (#132).
Three layers, bottom to top.

## Layer 1: one comparison is one number, d

Every metric row knows how to compare any two datasets and return one
number, d. The weights live here, inside a single d -- never in the
scoring loop above.

- **CD (raw contributions):** pool both sides' contributions, compute the
  earth mover's distance between them.
- **CB (round means):** per round, the absolute difference of the two
  means; average the 24 rounds equally. Rounds are equally populated by
  the game's design, so uniform weights (same for all C/S/P rows).
- **RCA (reaction by round type):** an EMD within each round type,
  averaged with the types' human presence rates (79% of rows are
  no-switch rounds, only 2% are chose-to-stay -- a rare bin should not
  count like a common one). Same idea for all R rows.

The weights are computed once from the full human data and reused in
every comparison, so every model is measured with the same yardstick.

## Layer 2: the problem -- a lone d means nothing

Say d(human, sim) = 1.6 for CD. Bad? There is nothing to compare it to.
And part of that 1.6 is not model error at all: 50 episodes is a small
sample, and if the human experiment were run twice, the two datasets
would not match either. Every row has a floor below which no model can
go, and the floor differs per row.

## Layer 3: measure the floor with humans vs humans

One repeat:

- shuffle the 50 human episodes, split into halves h_a and h_b (25 each)
- **floor piece:** d(h_a, h_b) -- how far two same-sized groups of real
  humans are apart. For CD this is about 1.08. A perfect model cannot
  beat this: h_b IS a perfect model of humans, and 1.08 is what it gets.
- **model piece:** d(h_a, s), with s = 25 random sim episodes. Same
  reference half, same sample size, so small-sample inflation hits both
  pieces equally and cancels.

One split is itself random, so repeat ~500 times with fresh shuffles
(one fixed seed, so every model sees identical splits and draws).
Average the model pieces, average the floor pieces, divide:

    score = typical human-vs-sim distance / typical human-vs-human distance

## Reading a score

"How many times farther from humans is this model than humans are from
themselves?"

- **~1** -- at the ceiling; indistinguishable from a second human sample.
  Below 1 happens when a sim is smoother than humans; read it as "at the
  ceiling" too.
- **1-2** -- minor deviation.
- **2-5** -- clear deviation.
- **> 5** -- not reproduced.

Examples from the linear-stack run (`22_2g8a_linear_self_ridge_contr`):

- PA for lin_ridge: raw d ~2.3 becomes **7.4** -- seven times outside
  natural human variation; the punishment distribution is simply wrong.
- PA for lin_multinomial: **0.63** -- at the ceiling.
- RCC (the ceiling-punishment reaction): the scariest raw number on the
  board (~7.0) becomes a mild **1.7**, because its own human floor is 4.4
  -- a contrast built on 44 events is naturally that unstable.

That is the point of the schema: raw d values are apples and oranges
across rows; scores are all in one unit, multiples of natural human
variation.
