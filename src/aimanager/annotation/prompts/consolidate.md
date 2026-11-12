# Task: consolidate a codebook into codes and families

Below is a list of codes, each with a name, label, definition and examples. $kind_text

Do two things, derived only from the codes themselves (no scheme is given):

1. `merges`: find codes that are the same code. Merge
   - codes that differ only in wording or in a minor variant (for example `x_contribution_trend` and `x_contribution_trend_2`),
   - codes that are the increase and the decrease variant of one lever (for example "raise the fee on Mondays" and "lower the fee on Mondays" become one code "change the fee on Mondays"; the direction is not part of the code).
   For each merge give the codes it joins (`codes`, at least two), the name that survives (`keep`, one of them), and a `label` and rewritten `definition` that covers all of them. Do not merge codes that are different mechanisms or reasons only because they are related; those belong in one family instead. Leave `merges` empty if no codes are the same.
2. `families`: group every code that survives the merges into a family, a broader grouping. Aim for roughly 5 to 10 families. Families must be mutually exclusive: every code is in exactly one family. For each family give a short snake_case `name`, a one-sentence `definition`, and its `codes` (use the surviving names).

Never define a code or a family as the absence of something ("does not ...", "avoids ...", "ignores ...").
