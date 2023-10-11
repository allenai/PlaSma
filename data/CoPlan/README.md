### Update:

This folder contains CoPlan dataset for the following 3 tasks:

1. **Goal-based Planning**
2. **Constrained Planning**
3. **Counterfactual Replanning**


Note that same data is used for task 2 and 3. The only difference is the input to the model. Task 2 only takes goal + condition to generate a plan, whereas task 3 takes goal + condition + (initial) plan to rewrite/update the plan.
 
This is CoPlan_v2 which contains Curie-generated, Davinci-generated and human-written plans for the original `goal-based planning` task. The paper used CoPlan_v1 which didn't include Davinci-generated plans.

There is no difference in v1 and v2 on the `constrained plannning` and `counterfactual replanning` tasks.


We additionally provided the processed data for multitask distillation which can be find in `multi-task` folder.


