# Task: judge which mechanisms a manager rule implements

Below is a codebook of manager mechanisms (grouped into families where the codebook has them; the families help to tell neighbouring codes apart), then the game setting and one manager rule (parameters with definitions and sweep ranges, optional constraints, and code). For every code decide whether the rule implements that mechanism.

For every code answer in this order: first `evidence`, then `present`, then `switchable_off`.

Guidelines:
- `evidence` comes first: look for the line of code or parameter definition that implements the code and copy it verbatim; leave it empty if there is none. `present` may be true only if `evidence` is not empty.
- `present` is true if the rule's code actually does what the definition says for some parameter values within the declared sweep ranges, even if another value in the range would switch it off. A parameter definition alone is not enough if the code does not use it that way.
- `switchable_off` is true if some parameter value inside the declared sweep range disables the mechanism entirely (for example a weight whose range includes 0); false if the mechanism acts for every value in the range, and false for an absent code.
- Judge each code independently; a rule may implement many codes or few.
