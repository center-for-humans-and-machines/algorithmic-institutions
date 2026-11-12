# Task: build a codebook of what reasons rest on

Below is a numbered list of short phrases, each saying, in the search agent's words, what one of its reasons rests on (an analysis, a result, an expectation, a calculation, a source of data). Each phrase is followed by a context line with the reason it supports and what the evidence points to (an exploration of these notes, an earlier version, nothing); the context line helps to understand the phrase and is not to be coded. Many phrases describe the same kind of basis in different words. Group them into a small codebook of distinct kinds of basis: a code is a kind of support a reason can rest on. No scheme is given; derive the codes from the items. Aim for about $target codes; when in doubt, merge rather than add.

Guidelines:
- A code is one kind of basis for a reason, at the level of abstraction of these examples from an unrelated domain (what a librarian's reasons rest on): "statistics from the library's own records", "a trial run at one branch", "what other libraries report", "the librarian's expectation". A code must be specific enough that one can decide for any phrase whether it fits.
- A code names the kind of basis, not a specific number, dataset or exploration: "a regression on last year's loans" and "a table of loans by fee size" are both statistics from the library's own records.
- Expect few codes. Codes should not overlap: a phrase should fit exactly one code. Merge near-duplicates.
- `name`: short snake_case identifier (not `none` or `failed`); `label`: a few words; `definition`: one or two sentences saying what counts; `fits`: two short examples that fit; `does_not_fit`: one or two near misses that belong elsewhere.
