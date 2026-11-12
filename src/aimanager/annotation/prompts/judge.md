# Task: judge which mechanisms a manager rule implements

Below is a codebook of manager mechanisms, then one manager rule (parameters with definitions, optional constraints, and code). For every code decide whether the rule implements that mechanism.

Guidelines:
- `present` is true only if the rule's code actually does what the definition says, for some parameter values in the rule's range. A parameter definition alone is not enough if the code does not use it that way.
- For a present code, `quote` is the line of code or parameter definition that implements it, copied verbatim. For an absent code, `quote` is an empty string.
- Judge each code independently; a rule may implement many codes or few.
