I think we have two main objectives:
Running a pilot with a minimum group size of 2
Training manager with objective to maintain group

I also had a look at the code an identified the following.
Fulfilled and not fulfilled conditions for training on group size:
* Group Selection is already in environment (good)
* Manager rewards are computed on the participant (bad; needs to be on the group)
* Manager assumes fully connected network during training and network is not stored in replay memory (bad)

Rough todos:
* Refactor to store rewards per manager (not per participant)
* Refactor flexible storage of network during playouts and while loading for training
* Validate that results do not change
* Implement network update during playouts
* Retrain on updated reward
* Maybe you can look first into the running another pilot with minimum group size.




# Strategy
* We update the interaction network for group members; but for the manager, we use fully connected


# Thoughts
* Manager punish group member in round t
* Member moves in different group in round t+1


# We mainly need to make sure that if
* a group member is entering a group, punishments from the other manager are not to be applied
*

        # We assuming a decomposition of the Q-value, in which the managers q value
        # can be decomposed into individual q values for the actions on each agent



Strategty 1
* First reshape rewards to agents during training (should work for fixed groups)
* Second, implement summing q values for reward (second step)


Strategy - enable dynamic groups
* We doublicate the input data by the number of groups
    * repeat observation to add group dimension
    * add in-out group identifier
    * reshape group into batch
    * compute q
    * reshape q to regain batch
* We add to the input information if a group member is in the group, or the other group
* We compute a single q-value for each group, agent and action
* We only further consider the q-value for the group in which the actual agent is positioned


Implementation Options
A:
* Doublicate in Manager
* Compute ingroup metric in Manager
* add metric to obs
* ...

B:




# TODO

[] Check if group and action is correct in regard to round, given the weired shift I am working with
