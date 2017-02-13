# behavioral-cloning
Drive a car using deep learning

# design-solution
As an engineer, I want to train a deep learning model, so as to drive a car around track 1. Design a solution that minimizes data used and model complexity. Use the scientific method to conduct experiments. Change 1 variable at a time to see whether the result confirms or rejects the hypothesis. Take into account hardware and time constraints.

Hardware
- Central processing unit: Intel Core 2 Duo 2.66 GHz
- Random access memory: 4 GB

# get-data
Use data provided by Udacity to minimize data collection time. Collect data while driving anti-clockwise around track 1 to mitigate right turn bias. Collect data while recovering from hitting a kerb.

Udacity
- Source: https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
- Images = 8036 * 3 (center, left, right) = 24108
- Size: 323 MB

Simulator
- 2017_02_12_17_51 to 2017_02_12_18_08
    - Images = 1994 * 3 (center, left, right) = 5982
    - Track 1, anti-clockwise, right side recovery
- 2017_02_12_18_52 to 2017_02_12_18_55
    - Images = 2068 * 3 (center, left, right) = 6204
    - Track 1, anti-clockwise, center
- 2017_02_12_21_30 to 2017_02_12_21_33
    - Images = 2121 * 3 (center, left, right) = 6363
    - Track 1, anti-clockwise, center
- 2017_02_13_08_00 to 2017_02_13_08_04
    - Images = 2024 * 3 (center, left, right) = 6072
    - Track 1, anti-clockwise, center
- 2017_02_13_19_07 to 2017_02_13_19_10
    - Images = 2034 * 3 (center, left, right) = 6102
    - Track 1, anti-clockwise, center
- 2017_02_13_20_19 to 2017_02_13_20_42
    - Images = 1622 * 3 (center, left, right) = 4866
    - Track 1, anti-clockwise, right and left recovery
- 2017_02_13_21_38 to 2017_02_13_21_41
    - Images = 1908 * 3 (center, left, right) = 5724
    - Track 1, anti-clockwise, center

Total samples = 21807
Total images = 65421

# design-model
Use the NVIDIA model as the base architecture.

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
Get all samples from the driving log. Split samples 80% / 20% into train and validation set to test if model is over fitting. Shuffle samples to reduce order bias. Flip image to generate more samples and reduce left and right turn bias. Initial batch size of 32 is too large as images are unable to fit in memory. Use batch size of 2 to increase samples per second and fit images in memory.

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
- Train set size: 6428
- Learning rate: 0.001
- Epoch: 2
- Training time: 471 s
- Samples per second: 27.3
- Track 1 performance: Car drifts right. Car hits kerb.

Experiment 25
- Image: Center, left, right, color, vertical crop, normalized, centered
- Train set size: 19284
- Learning rate: 0.001
- Epoch: 2
- Training time: 1590 s
- Samples per second: 24.3
- Track 1 performance: Car drifts right. Car hits kerb.

Experiment 26
- Image: Center, flip, color, vertical crop, normalized, centered
- Train set size: 9925 * 2 = 19850
- Learning rate: 0.001
- Epoch: 2
- Training time: 745 s
- Samples per second: 53.3
- Track 1 performance: Car drifts right. Car hits kerb.

Experiment 27
- Image: Center, flip, color, vertical crop, normalized, centered
- Train set size: 9678 * 2 = 19356
- Learning rate: 0.001
- Epoch: 2
- Training time: 738 s
- Samples per second: 52.5
- Track 1 performance: Car drifts right. Car hits kerb.

Experiment 28
- Image: Center, flip, color, vertical crop, normalized, centered
- Train set size: 9678 * 2 = 19356
- Learning rate: 0.000001
- Epoch: 2
- Training time: 556 s
- Samples per second: 69.6
- Track 1 performance: Goes straight, drifts right, hits kerb

Experiment 29
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 9678 * 6 = 58068
- Learning rate: 0.000001
- Epoch: 2
- Training time: 544 s
- Samples per second: 213
- Track 1 performance: Turns right, hits kerb

Experiment 30
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 9678 * 6 = 58068
- Learning rate: 0.000001
- Epoch: 2
- Training time: 544 s
- Samples per second: 213
- Track 1 performance: Turns right, hits kerb

Experiment 31
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 11375 * 6 = 68250
- Learning rate: 0.000001
- Epoch: 2
- Training time: 626 s
- Samples per second: 218
- Track 1 performance: Turns right, hits kerb

Experiment 32
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 11375 * 6 = 68250
- Learning rate: 0.000001
- Epoch: 2
- Training time: 755 s
- Samples per second: 181
- Track 1 performance: Drifts right, drives on right double yellow line, drives on red and white rumble strips, hits kerb

Experiment 33
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 11375 * 6 = 68250
- Learning rate: 0.00001
- Epoch: 2
- Training time: 752 s
- Samples per second: 182
- Track 1 performance: Drifts right, hits kerb

Experiment 34
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 11375 * 6 = 68250
- Learning rate: 0.000001
- Epoch: 8
- Training time: 375 s
- Samples per second: 182
- Track 1 performance: Drifts right, drives on red and white rumble strips, hits kerb

Experiment 35
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 11375 * 6 = 68250
- Learning rate: 0.000001
- Epoch: 16
- Training time: 377 s
- Samples per second: 181
- Track 1 performance: Drifts right, drives on right double yellow line, drives on red and white rumble strips, hits kerb

Experiment 36
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 11375 * 6 = 68250
- Learning rate: 0.000001
- Epoch: 4
- Training time: 389 s
- Samples per second: 175
- Track 1 performance: Drifts right, drives on double yellow line, drives on red and white rumble strips, drives on to bridge, turns left, hits wall

Experiment 37
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 12994 * 6 = 77964
- Learning rate: 0.000001
- Epoch: 4
- Training time: 444 s
- Samples per second: 176
- Track 1 performance: Turns right, hits kerb

Experiment 38
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 12994 * 6 = 77964
- Learning rate: 0.000001
- Epoch: 8
- Training time: 444 s
- Samples per second: 176
- Track 1 performance: Drifts right, drives on double yellow line, stops on right side

Experiment 39
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 14621 * 6 = 87726
- Learning rate: 0.000001
- Epoch: 8
- Training time: 507 s
- Samples per second: 174
- Track 1 performance: Goes straight, drifts right, drives on red and white rumble strips, turns right, hits kerb

Experiment 40
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 15919 * 6 = 95514
- Learning rate: 0.000001
- Epoch: 8
- Training time: 555 s
- Samples per second: 172
- Track 1 performance: Goes straight, drifts right, turn right, hit kerb

Experiment 41
- Image: Center, left, right, flip left, flip right, color, vertical crop, normalized, centered
- Train set size: 15919 * 5 = 79595
- Learning rate: 0.000001
- Epoch: 8
- Training time: 557 s
- Samples per second: 143
- Track 1 performance: Drift right, hit kerb

Experiment
- Image: 
- Train set size: 
- Learning rate: 
- Epoch: 
- Training time:  s
- Samples per second: 
- Track 1 performance: 

# reflect