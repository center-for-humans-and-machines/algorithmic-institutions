# Task: describe the mechanisms of one manager rule

Below is one manager rule: its parameters with their definitions, optional constraints, and the code that computes the punishment. List the distinct mechanisms the rule uses, at the level of *what the manager does to players*, for example "punishes a member in proportion to how far they fall below the group's mean contribution" or "stops punishing after a given round".

Guidelines:
- Describe behaviour towards players (who is punished, when, how much, depending on what), not implementation details (tensor operations, clamping, numerical safeguards, variable names).
- One entry per mechanism. A parameter that only scales another mechanism is part of that mechanism, not a mechanism of its own; a condition that switches behaviour on or off (a round window, a group-size gate, a newcomer clause) is its own mechanism.
- Write each description as one plain sentence that someone who has not seen the code understands. Use general terms, not this rule's parameter names.
- For each mechanism give a quote: the line of code or the parameter definition that most directly implements it, copied verbatim.
- Include every mechanism the rule actually implements, and nothing it does not implement. Typically a rule has two to eight mechanisms.
