# Task: build a codebook of manager mechanisms

Below is a numbered list of short descriptions of mechanisms that manager rules use (what a manager does to players). Many describe the same mechanism in different words. Group them into a codebook of distinct mechanisms, derived from the items.

Guidelines:
- A code is one mechanism at the level of what the manager does to players. For the level of abstraction, examples from an unrelated domain (a library's rules for overdue books): "fee grows with the days overdue", "borrowing suspended after repeated lateness", "grace period before any fee". A code must be specific enough that one can decide for any rule whether it implements the code.
- A code names the mechanism, not an implementation detail or a parameter value: different rates or lengths of one idea are one code ("a fee of 0.20 per day" and "a fee of 0.50 per day" are both "fee grows with the days overdue").
- Codes should not overlap: an item should fit exactly one code. Prefer merging near-duplicates over splitting by minor variants.
- A code is something a manager does. Never define a code as the absence of a mechanism ("does not ...", "avoids ...", "ignores ..."): such an item describes a rejected idea, and its code is the idea itself. Decisions about the search or the rule's parameters (removing, fixing or keeping a parameter) are not mechanisms and get no code.
- Cover the frequent mechanisms; rare idiosyncratic items may stay without a code.
- `name`: short snake_case identifier (not `none` or `failed`); `label`: a few words; `definition`: one or two sentences saying what counts; `fits`: two short examples that fit; `does_not_fit`: one or two near misses that belong elsewhere.
