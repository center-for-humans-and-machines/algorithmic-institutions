# Task: build a codebook of reasons

Below is a numbered list of reasons the search agent gave for or against manager mechanisms. Each item is the reason itself, followed by a context line naming the mechanism it refers to and whether it argues for or against it; the context line is there to make the reason understandable and is not to be coded. Many items state the same reason in different words. Group them into a codebook of distinct kinds of reasons. No scheme is given; derive the codes from the items. Aim for about $target codes; when in doubt, merge rather than add.

Guidelines:
- A code is a kind of justification that could be given for several different mechanisms. It is not a specific claim with its numbers, and it is not tied to one mechanism. For the level of abstraction, examples from an unrelated domain (reasons a librarian gives for overdue rules): "fees make borrowers return books sooner", "harsh fees drive members away", "collecting fees costs staff time", "other libraries do it". A code must be specific enough that one can decide for any stated reason whether it fits.
- Group by the kind of justification, not by the mechanism: the same kind of reason given for different mechanisms is one code, and different reasons for one mechanism are different codes. Do not split by stance either.
- A code names the kind of reason, not a specific number or result: "a 0.20 fee halved late returns" and "fees reduce late returns" fit the same code.
- Codes should not overlap: an item should fit exactly one code. Merge near-duplicates.
- A code names what a reason claims. A claim that something is absent is a valid reason ("late fees bring no extra returns"); name the claim itself.
- Cover the frequent reasons; rare idiosyncratic items may stay without a code.
- `name`: short snake_case identifier (not `none` or `failed`); `label`: a few words; `definition`: one or two sentences saying what counts; `fits`: two short examples that fit; `does_not_fit`: one or two near misses that belong elsewhere.
