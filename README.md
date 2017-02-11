# behavioral-cloning
Drive a car using deep learning

# design-solution
Hardware
- Central processing unit: Intel Core 2 Duo 2.66 GHz
- Random access memory: 4 GB

# get-data

# design-model

# train-model

# evaluate-model
Rubric: https://review.udacity.com/#!/rubrics/432/view

Number|Image|Set size|Learning rate|Epoch|Training time|Samples per second|Loss|Notes
------|-----|--------|-------------|-----|-------------|------------------|----|-----
1|Center, normalized|256|0.01|4|1 min 7 s|15.3|0.0194|Model may be overfitting as difference between training and validation loss increases per epoch. Loss becomes not a number when training model again.
2|Center, normalized|3|0.000001|4|1 s|12|0.9166|Loss no longer becomes not a number due to reduced learning rate. Model predicts steering direction correctly.
3|Center, normalized|256|0.000001|4|1 min 6 s|15.5|0.0533|Loss plateaus. Validation loss is greater than training loss. Car makes a hard left turn.
4|Center, normalized|512|0.000001|4|2 min 14 s|15.3|0.0258|Loss plateaus. Training loss is greater than validation loss. Car makes a hard right turn.
5|Center, normalized|512|0.000001|4|2 min 22 s|14.4|0.0266|Validation loss is greater than training loss. Car makes a hard right turn with brief hard left turns.

Switched to fit_generator after encountering out of memory error with 1024 set size.

Number|Image|Samples per epoch|Learning rate|Epoch|Training time|Samples per second|Loss|Notes
------|-----|-----------------|-------------|-----|-------------|------------------|----|-----
6|Center|2|0.000001|4|1 s|16|132.5945|Training loss is greater than validation loss. Car makes hard left turn. Car hits the kerb. Car makes hard right turn.
7|Center|512|0.000001|4|9 min 34 s|4.28|2692.1430|Validation loss is greater than training loss. Car makes a hard right turn with brief hard left turns.
8|Center, normalized|2|0.000001|4|1 s|16|0.0271|Training loss is greater than validation loss. Car drifts to the right. Car hits the kerb.
9|Center, grayscale, normalized|2|0.000001|4|3 s|5|0.0062|Validation loss is greater than training loss. Car drifts to the right. Car hits the kerb.
10|Center, crop, grayscale, normalized|2|0.000001|4|1 s|16|0.1328|Training loss is greater than validation loss. Car drifts to the right. Car hits the kerb.
11|Center, crop, grayscale, normalized|512|0.000001|4|52 s|47.2|0.0114|Training loss is greater than validation loss. Car stays in lane with double yellow lines. Car does not turn left when lane markers change to red and white rumble strips. Car goes over the kerb. Car goes straight.
12|Center, crop, grayscale, normalized|1024|0.000001|4|1 min 46 s|46.4|0.0310|Validation loss is greater than training loss. Car goes straight. Car drifts to the right. Car goes over the kerb.
13|Center, left, right, crop, grayscale, normalized|1023|0.000001|4|1 min 9 s|59.3|0.0321|Training loss is greater than validation loss. Car drifts to the right. Car drives in between the right lane line and kerb. Car goes over the kerb.
14|Center, left, right, crop, grayscale, normalized|3069|0.000001|4|3 min 25 s|59.9|0.0561|Validation loss is greater than training loss. Car goes straight. Car drifts to the left. Car drives in between the left kerb and lane line. Car does not turn left when lane markers change to red and white rumble strips. Car goes over the kerb.
15|Center, left, right, crop, grayscale, normalized|3069|0.000001|2|1 min 43 s|59.6|0.0372|Validation loss is greater than training loss. Car goes straight. Car drifts to the right. Car drives in between the right lane line and kerb. Car goes over the kerb.
15|Center, left, right, crop, grayscale, normalized|9642|0.000001|2|5 min 40 s|56.7|0.0517|Validation loss is greater than training loss. Car drifts to the left. Car goes over the kerb.
16|Center, left, right, vertical crop, grayscale, normalized|9642|0.000001|2|8 min 45 s|36.7|0.0362|Training loss is greater than validation loss. Car goes straight. Car drifts to the left. Car turns left when lane markers change to red and white rumble strips. Car drifts left. Car goes over kerb.

# reflect