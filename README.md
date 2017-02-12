# behavioral-cloning
Drive a car using deep learning

# design-solution
As an engineer, I want to train a deep learning model, so as to drive a car around track 1. The solution should minimize data used and model complexity. The approach should take into account hardware and time constraints.

Hardware
- Central processing unit: Intel Core 2 Duo 2.66 GHz
- Random access memory: 4 GB

# get-data
Udacity
- Source: https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
- Images = 8036 * 3 (center, left, right) = 24108
- Size: 323 MB

Simulator
- 2017_02_12_17_32 to 2017_02_12_17_36
    - Images = 2377 * 3 (center, left, right) = 7131
    - Track 1, clockwise, center
- 2017_02_12_17_51 to 2017_02_12_18_08
    - Images = 1994 * 3 (center, left, right) = 5982
    - Track 1, anti-clockwise, right side recovery

# design-model
The base architecture is derived from the NVIDIA model.

NVIDIA model
- Source: https://arxiv.org/pdf/1604.07316v1.pdf

Layers
- Cropping2D: Crop sky and car
- Lambda: Normalize and center data
- Convolution2D: Extract features from image
    - Filter: 24
    - Kernel size: 5
    - Stride size: 2
- Activation: Use rectified linear unit to introduce non-linearity
- Dropout: Reduce over fitting
- Convolution2D: Extract features from image
    - Filter: 36
    - Kernel size: 5
    - Stride size: 2
- Activation: Use rectified linear unit to introduce non-linearity
- Dropout: Reduce over fitting
- Convolution2D: Extract features from image
    - Filter: 48
    - Kernel size: 5
    - Stride size: 2
- Activation: Use rectified linear unit to introduce non-linearity
- Dropout: Reduce over fitting
- Convolution2D: Extract features from image
    - Filter: 64
    - Kernel size: 3
- Activation: Use rectified linear unit to introduce non-linearity
- Dropout: Reduce over fitting
- Convolution2D: Extract features from image
    - Filter: 64
    - Kernel size: 3
- Activation: Use rectified linear unit to introduce non-linearity
- Dropout: Reduce over fitting
- Flatten: Reduce dimensionality
- Dense: Fully connect each node
    - Connections: 100
- Dense
    - Connections: 50
- Dense
    - Connections: 10
- Dense
    - Connections: 1

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
17|Center, left, right, flipped, vertical crop, grayscale, normalized|19284|0.000001|2|15 min 14 s|42.2|0.0304|Training loss is greater than validation loss. Car goes straight. Car turns left when lane markers change to red and white rumble strips. Car turns left when lane markers change to double yellow lines. Car crashes into left wall at the start of the bridge that goes over the water.
18|Center, left, right, flipped, grayscale, vertical crop, normalized, centered|19284|0.000001|2|15 min 10 s|42.4|0.0302|Training loss is greater than validation loss. Car goes straight. Car turns left when lane markers change to red and white rumble strips. Car turns left when lane markers change to double yellow lines. Car crashes into left wall in the middle of the bridge that goes over the water.
19|Center, left, right, flipped, grayscale, vertical crop, normalized, centered|9911|0.000001|2|7 min 44 s|42.7|0.0177|Training loss is greater than validation loss. Car goes straight. Car turns left when lane markers change to red and white rumble strips. Car turns left when lane markers change to double yellow lines. Car crashes into right wall in the middle of the bridge that goes over the water.
20|Center, left, right, flipped, grayscale, vertical crop, normalized, centered|59466|0.000001|2|47 min 1 s|42.2|0.0017|Training loss is greater than validation loss. Car goes straight. Car drifts right. Car goes over the kerb.
21|Center, left, right, flipped, grayscale, vertical crop, normalized, centered|63900|0.000001|2|1 h 8 min 42 s|31.0|0.0010|Training loss is greater than validation loss. Car does not turn left when lane markers change to red and white rumble strips. Car goes over the kerb.
22|Center, left, right, flipped, grayscale, vertical crop, normalized, centered|38568|0.000001|2|41 min 56 s|30.7|0.0049|Training loss is greater than validation loss. Car does not turn left when lane markers change to red and white rumble strips. Car goes over the kerb.
23|Center, left, right, flipped, grayscale, vertical crop, normalized, centered|38568|0.001|2|42 min 13 s|30.5|0.0000480|Training loss is greater than validation loss. Car goes straight. Car drifts right. Car ges over kerb.

Experiment 24
- Image: Center, color, vertical crop, normalized, centered
- Training set size: 6428
- Epoch: 2
- Training time: 471 s
- Samples per second: 27.3
- Track 1 performance: Car drifts right. Car hits kerb.

Experiment 25
- Image: Center, left, right, color, vertical crop, normalized, centered
- Training set size: 19284
- Epoch: 2
- Training time: 1590 s
- Samples per second: 24.3
- Track 1 performance: Car drifts right. Car hits kerb.

Experiment
- Image:
- Training set size:
- Epoch:
- Training time:
- Samples per second:
- Track 1 performance:

# reflect