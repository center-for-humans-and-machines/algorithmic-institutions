# Task: describe the mechanisms of one manager rule

Below, after the game setting, is one manager rule: its parameters with their definitions and sweep ranges, optional constraints, and the code that computes the punishment. List the distinct mechanisms the rule uses, at the level of *what the manager does to players*. Work bottom-up from this rule; there is no list of expected mechanisms.

The right level of abstraction, shown with an unrelated domain (a library's rules for overdue books):
- Mechanisms: "charges a fee that grows with the number of days a book is overdue"; "suspends borrowing for members who have several overdue books"; "charges nothing during a grace period after the due date".
- Implementation details, not mechanisms: "rounds the fee to whole cents"; "stores the due date as a timestamp"; "clips the computed value to avoid negative fees".
- Not separate mechanisms: "the fee per day is 0.20" and "a parameter scales the daily fee" both belong to the fee mechanism.

Guidelines:
- Describe behaviour towards players (who is affected, when, how much, depending on what), not implementation details (tensor operations, clamping, numerical safeguards, variable names).
- One entry per mechanism. A parameter that only scales another mechanism is part of that mechanism, not a mechanism of its own; a condition that switches behaviour on or off (a time window, an eligibility condition) is its own mechanism.
- Write each description as one plain sentence that someone who has not seen the code understands. Use general terms, not this rule's parameter names.
- For each mechanism give a quote: the line of code or the parameter definition that most directly implements it, copied verbatim.
- Include every mechanism the rule actually implements, and nothing it does not implement. Typically a rule has two to eight mechanisms.
