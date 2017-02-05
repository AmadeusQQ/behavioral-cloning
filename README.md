# behavioral-cloning
Drive a car using deep learning

# design-solution

# get-data

# design-model

# train-model

# evaluate-model
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

# reflect