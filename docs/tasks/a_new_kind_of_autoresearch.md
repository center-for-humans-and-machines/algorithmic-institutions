This is about the policy finder for experiment 2: Finding a policy that does well regarding common good in its group in a group switching setup where it plays another governor.

Before, we compared every policy only against ah and chose the optimal policy based on the score against it. Next up, we want to try to optimize a more general policy that does well against various policies. We will then expect good learned rules to adapt to the policy that the other governor is employing.

As optimization target, I am thinking about the following receipe:
An average of 4 scores: Common good against ah, common good against zero, common good against levins rule and common good against the previous best policy found with the policy finder.

Governors should not be informed who their opponent is in simulation (they only see their punishment behavior)

Make sure that in the policy finder artifact we are separating the four scores against the four possible opponent models.