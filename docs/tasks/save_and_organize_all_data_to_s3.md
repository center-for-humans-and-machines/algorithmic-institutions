# Cleanly save all our data to S3
So in this project we have some important data and models that are all a little messy currently.

We want to cleanly save them in a structured way in S3. Especially, we want to long-term move away from git-lfs and have our large files in S3 instead.

Especially we want to save right now:
1. Experiment 1 models (The ones actually used in the experiment with human participants)
    * Behavioral clone models
    * RL policy models
2. Pre-Experiment 1 human only data. Collected single human manager + 4 human players data that predates Experiment 1. (The data the experiment 1 behavioral clone was trained on)
3. Experiment 1 data.
    * The human-AI interaction data collected in experiment 1.
    * Also survey answers and stuff.
4. The two human manager group switching data. This is the data the human clones for experiment 2 are trained on.
5. Everything relevant for the autoresearch pipeline on the human clones.
    * Especially including the final human clone models: Contribution, switching and punishment.
    
Can you find everything from that list? Are there things that seem like they should be saved but I didn't include them in the list? Is there anything that seems important that is mentioned but you can't find it? Ask me about those!

Then, after confirming everything we have, please store all of this in the S3 bucket. Come up with a good folder / organization structure yourself. Right now, do it alongside what is already on the bucket and let me know what is not duplicate in the bucket in the end.