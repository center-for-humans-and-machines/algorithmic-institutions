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