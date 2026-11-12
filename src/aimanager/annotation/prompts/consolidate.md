# Task: consolidate a codebook into codes and families

Below is a list of codes, each with a name, label, definition and examples. $kind_text

Do two things, derived only from the codes themselves (no scheme is given):

1. `merges`: find codes that are the same code. Merge two codes only if they describe the same thing: anything that fits one necessarily fits the other, and the other way round (they differ only in wording or a minor variant, for example `x_contribution_trend` and `x_contribution_trend_2`). If you can name a case that fits one code but not the other, do not merge them.
   Never merge codes that refer to different things a rule conditions on or acts on (for example a fee that depends on the number of books borrowed and a fee that depends on the borrower's annual budget are different codes, even though both scale the fee). Related codes belong in one family, not in one code.
   For each merge give the codes it joins (`codes`, at least two), the `reason` why they are one code (why anything that fits one fits the other and the other way round, in one sentence), the name that survives (`keep`, one of them), and a `label` and rewritten `definition` that covers all of them. Leave `merges` empty if no codes are the same.
2. `families`: group every code that survives the merges into a family, a broader grouping. Aim for roughly 5 to 10 families, each with at least two codes where possible. Families must be mutually exclusive, and every surviving code must be in exactly one family: leave no code out. No catch-all family: a family's definition must say what its codes have in common that the codes of other families lack (not "other", "general" or "structural constraints on punishment" that any code would fit). For each family give a short snake_case `name`, a one-sentence `definition`, and its `codes` (use the surviving names).

$kind_rules
