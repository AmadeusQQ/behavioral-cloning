# behavioral-cloning
Drive a car using deep learning

# design-solution
As an engineer, I want to train a deep learning model, so as to drive a car around track 1.

Design a solution that minimizes data used and model complexity. Use the scientific method to conduct experiments and test whether the result confirms or rejects the hypothesis. Take into account software and hardware constraints.

Software
- Operating system: Ubuntu 16.04
    - Memory leaks over time

Hardware
- Central processing unit: Intel Core 2 Duo 2.66 GHz
- Random access memory: 4 GB

# get-data
Collect data while driving clockwise and anti-clockwise to reduce turn bias. Collect data while recovering from hitting a kerb and drifting to the side.

2017-02-16-center-1
- Track: 1
- Type: Center
- Direction: Counter-clockwise
- Laps: 2
- Frames: 5648
- Images = 16944

2017-02-16-center-2
- Track: 1
- Type: Center
- Direction: Clockwise
- Laps: 2
- Frames: 7033
- Images = 21096

2017-02-16-recovery-1
- Track: 1
- Type: Recovery
- Direction: Counter-clockwise
- Laps: 1
- Frames: 1081
- Images = 3243

# design-model
Implement the NVIDIA model as the base architecture. Take in a (160, 320, 3) color image as input and output the steering angle as a float.

NVIDIA model
- Source: https://arxiv.org/pdf/1604.07316v1.pdf

Layers
- Cropping2D: Crop sky and car
- Lambda: Normalize and center data
- Convolution2D: Extract features from image
    - Filter: 24
    - Kernel size: 5
    - Stride size: 2
- Dropout: Reduce over fitting
- Convolution2D: Extract features from image
    - Filter: 36
    - Kernel size: 5
    - Stride size: 2
- Dropout: Reduce over fitting
- Convolution2D: Extract features from image
    - Filter: 48
    - Kernel size: 5
    - Stride size: 2
- Dropout: Reduce over fitting
- Convolution2D: Extract features from image
    - Filter: 64
    - Kernel size: 3
- Dropout: Reduce over fitting
- Convolution2D: Extract features from image
    - Filter: 64
    - Kernel size: 3
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
Get all samples from the driving log. Split samples 80% / 20% into train and validation set to test if model is over fitting. Shuffle samples to reduce order bias. Flip left and right images to generate more samples and reduce left and right turn bias. Batch size of 32 is too large as images are unable to fit in memory. Use batch size of 16 based on hardware constraints.

# evaluate-model
Rubric: https://review.udacity.com/#!/rubrics/432/view

**Bold** values denote change from previous experiment.

**Bold** experiments denote key discoveries.

Experiment 1
- Image: Center, normalized
- Set size: 256
- Learning rate: 1e-2
- Epoch: 4
- Training time: 67 s
- Samples per second: 15.3
- Loss: 0.0194
- Notes: Model may be overfitting as difference between training and validation loss increases per epoch. Loss becomes not a number when training model again.

**Experiment 2**
- Image: Center, normalized
- Set size: **3**
- Learning rate: **1e-6**
- Epoch: 4
- Training time: 1 s
- Samples per second: 12
- Loss: 0.9166
- Notes: Loss no longer becomes not a number due to reduced learning rate. Model predicts steering direction correctly.

Experiment 3
- Image: Center, normalized
- Set size: **256**
- Learning rate: 1e-6
- Epoch: 4
- Training time: 66 s
- Samples per second: 15.5
- Loss: 0.0533
- Notes: Loss plateaus. Validation loss is greater than training loss. Car makes a hard left turn.

Experiment 4
- Image: Center, normalized
- Set size: **512**
- Learning rate: 1e-6
- Epoch: 4
- Training time: 134 s
- Samples per second: 15.3
- Loss: 0.0258
- Notes: Loss plateaus. Training loss is greater than validation loss. Car makes a hard right turn.

**Change experiment format. Switch to fit_generator after encountering out of memory error with 1024 set size.**

Experiment 5
- Image: Center
- Samples per epoch: 2
- Learning rate: 1e-6
- Epoch: 4
- Training time: 1 s
- Samples per second: 16
- Loss: 132.5945
- Notes: Training loss is greater than validation loss. Car makes hard left turn. Car hits the kerb. Car makes hard right turn.

Experiment 6
- Image: Center
- Samples per epoch: **512**
- Learning rate: 1e-6
- Epoch: 4
- Training time: 574 s
- Samples per second: 4.28
- Loss: 2692.1430
- Notes: Validation loss is greater than training loss. Car makes a hard right turn with brief hard left turns.

Experiment 7
- Image: Center, **normalized**
- Samples per epoch: **2**
- Learning rate: 1e-6
- Epoch: 4
- Training time: 1 s
- Samples per second: 16
- Loss: 0.0271
- Notes: Training loss is greater than validation loss. Car drifts to the right. Car hits the kerb.

Experiment 8
- Image: Center, **grayscale**, normalized
- Samples per epoch: 2
- Learning rate: 1e-6
- Epoch: 4
- Training time: 3 s
- Samples per second: 5
- Loss: 0.0062
- Notes: Validation loss is greater than training loss. Car drifts to the right. Car hits the kerb.

Experiment 9
- Image: Center, **crop**, grayscale, normalized
- Samples per epoch: 2
- Learning rate: 1e-6
- Epoch: 4
- Training time: 1 s
- Samples per second: 16
- Loss: 0.1328
- Notes: Training loss is greater than validation loss. Car drifts to the right. Car hits the kerb.

Experiment 10
- Image: Center, crop, grayscale, normalized
- Samples per epoch: **512**
- Learning rate: 1e-6
- Epoch: 4
- Training time: 52 s
- Samples per second: 47.2
- Loss: 0.0114
- Notes: Training loss is greater than validation loss. Car stays in lane with double yellow lines. Car does not turn left when lane markers change to red and white rumble strips. Car goes over the 
kerb. Car goes straight.

Experiment 11
- Image: Center, crop, grayscale, normalized
- Samples per epoch: **1024**
- Learning rate: 1e-6
- Epoch: 4
- Training time: 106 s
- Samples per second: 46.4
- Loss: 0.0310
- Notes: Validation loss is greater than training loss. Car goes straight. Car drifts to the right. Car goes over the kerb.

Experiment 12
- Image: Center, **left**, **right**, crop, grayscale, normalized
- Samples per epoch: **1023**
- Learning rate: 1e-6
- Epoch: 4
- Training time: 69 s
- Samples per second: 59.3
- Loss: 0.0321
- Notes: Training loss is greater than validation loss. Car drifts to the right. Car drives in between the right lane line and kerb. Car goes over the kerb.

Experiment 13
- Image: Center, left, right, crop, grayscale, normalized
- Samples per epoch: **3069**
- Learning rate: 1e-6
- Epoch: 4
- Training time: 205 s
- Samples per second: 59.9
- Loss: 0.0561
- Notes: Validation loss is greater than training loss. Car goes straight. Car drifts to the left. Car drives in between the left kerb and lane line. Car does not turn left when lane markers change to 
red and white rumble strips. Car goes over the kerb.

Experiment 14
- Image: Center, left, right, crop, grayscale, normalized
- Samples per epoch: 3069
- Learning rate: 1e-6
- Epoch: **2**
- Training time: 103 s
- Samples per second: 59.6
- Loss: 0.0372
- Notes: Validation loss is greater than training loss. Car goes straight. Car drifts to the right. Car drives in between the right lane line and kerb. Car goes over the kerb.

Experiment 15
- Image: Center, left, right, crop, grayscale, normalized
- Samples per epoch: **9642**
- Learning rate: 1e-6
- Epoch: 2
- Training time: 340 s
- Samples per second: 56.7
- Loss: 0.0517
- Notes: Validation loss is greater than training loss. Car drifts to the left. Car goes over the kerb.

**Experiment 16**
- Image: Center, left, right, **vertical crop**, grayscale, normalized
- Samples per epoch: 9642
- Learning rate: 1e-6
- Epoch: 2
- Training time: 525 s
- Samples per second: 36.7
- Loss: 0.0362
- Notes: Training loss is greater than validation loss. Car goes straight. Car drifts to the left. Car turns left when lane markers change to red and white rumble strips. Car drifts left. Car goes 
over kerb.

Experiment 17
- Image: Center, left, right, flipped, vertical crop, grayscale, normalized
- Samples per epoch: **19284**
- Learning rate: 1e-6
- Epoch: 2
- Training time: 914 s
- Samples per second: 42.2
- Loss: 0.0304
- Notes: Training loss is greater than validation loss. Car goes straight. Car turns left when lane markers change to red and white rumble strips. Car turns left when lane markers change to double yellow lines. Car crashes into left wall at the start of the bridge that goes over the water.

**Experiment 18**
- Image: Center, left, right, flipped, grayscale, vertical crop, normalized, **centered**
- Samples per epoch: 19284
- Learning rate: 1e-6
- Epoch: 2
- Training time: 910 s
- Samples per second: 42.4
- Loss: 0.0302
- Notes: Training loss is greater than validation loss. Car goes straight. Car turns left when lane markers change to red and white rumble strips. Car turns left when lane markers change to double yellow lines. Car crashes into left wall in the middle of the bridge that goes over the water.

Experiment 19
- Image: Center, left, right, flipped, grayscale, vertical crop, normalized, centered
- Samples per epoch: **9911**
- Learning rate: 1e-6
- Epoch: 2
- Training time: 464 s
- Samples per second: 42.7
- Loss: 0.0177
- Notes: Training loss is greater than validation loss. Car goes straight. Car turns left when lane markers change to red and white rumble strips. Car turns left when lane markers change to double yellow lines. Car crashes into right wall in the middle of the bridge that goes over the water.

Experiment 20
- Image: Center, left, right, flipped, grayscale, vertical crop, normalized, centered
- Samples per epoch: **59466**
- Learning rate: 1e-6
- Epoch: 2
- Training time: 2821 s
- Samples per second: 42.2
- Loss: 0.0017
- Notes: Training loss is greater than validation loss. Car goes straight. Car drifts right. Car goes over the kerb.

Experiment 21
- Image: Center, left, right, flipped, grayscale, vertical crop, normalized, centered
- Samples per epoch: **63900**
- Learning rate: 1e-6
- Epoch: 2
- Training time: 4122 s
- Samples per second: 31.0
- Loss: 0.0010
- Notes: Training loss is greater than validation loss. Car does not turn left when lane markers change to red and white rumble strips. Car goes over the kerb.

Experiment 22
- Image: Center, left, right, flipped, grayscale, vertical crop, normalized, centered
- Samples per epoch: **38568**
- Learning rate: 1e-6
- Epoch: 2
- Training time: 2516 s
- Samples per second: 30.7
- Loss: 0.0049
- Notes: Training loss is greater than validation loss. Car does not turn left when lane markers change to red and white rumble strips. Car goes over the kerb.

Experiment 23
- Image: Center, left, right, flipped, grayscale, vertical crop, normalized, centered
- Samples per epoch: 38568
- Learning rate: **0.001**
- Epoch: 2
- Training time: 2533 s
- Samples per second: 30.5
- Loss: 0.0000480
- Notes: Training loss is greater than validation loss. Car goes straight. Car drifts right. Car ges over kerb.

Experiment 24
- Image: Center, **color**, vertical crop, normalized, centered
- Train set size: **6428**
- Learning rate: 0.001
- Epoch: 2
- Training time: 471 s
- Samples per second: 27.3
- Track 1 performance: Car drifts right. Car hits kerb.

Experiment 25
- Image: Center, **left**, **right**, color, vertical crop, normalized, centered
- Train set size: **19284**
- Learning rate: 0.001
- Epoch: 2
- Training time: 1590 s
- Samples per second: 24.3
- Track 1 performance: Car drifts right. Car hits kerb.

Experiment 26
- Image: Center, **flip**, color, vertical crop, normalized, centered
- Train set size: 9925 * 2 = **19850**
- Learning rate: 0.001
- Epoch: 2
- Training time: 745 s
- Samples per second: 53.3
- Track 1 performance: Car drifts right. Car hits kerb.

Experiment 27
- Image: Center, flip, color, vertical crop, normalized, centered
- Train set size: 9678 * 2 = **19356**
- Learning rate: 1e-3
- Epoch: 2
- Training time: 738 s
- Samples per second: 52.5
- Track 1 performance: Car drifts right. Car hits kerb.

**Experiment 28**
- Image: Center, flip, color, vertical crop, normalized, centered
- Train set size: 9678 * 2 = 19356
- Learning rate: **1e-6**
- Epoch: 2
- Training time: 556 s
- Samples per second: 69.6
- Track 1 performance: Goes straight, drifts right, hits kerb

Experiment 29
- Image: Center, **left**, **right**, flip, color, vertical crop, normalized, centered
- Train set size: 9678 * 6 = **58068**
- Learning rate: 1e-6
- Epoch: 2
- Training time: 544 s
- Samples per second: 213
- Track 1 performance: Turns right, hits kerb

Experiment 30
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 11375 * 6 = **68250**
- Learning rate: 1e-6
- Epoch: 2
- Training time: 626 s
- Samples per second: 218
- Track 1 performance: Turns right, hits kerb

Experiment 31
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 11375 * 6 = 68250
- Learning rate: **1e-5**
- Epoch: 2
- Training time: 752 s
- Samples per second: 182
- Track 1 performance: Drifts right, hits kerb

Experiment 32
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 11375 * 6 = 68250
- Learning rate: **1e-6**
- Epoch: **8**
- Training time: 375 s
- Samples per second: 182
- Track 1 performance: Drifts right, drives on red and white rumble strips, hits kerb

Experiment 33
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 11375 * 6 = 68250
- Learning rate: 1e-6
- Epoch: **16**
- Training time: 377 s
- Samples per second: 181
- Track 1 performance: Drifts right, drives on right double yellow line, drives on red and white rumble strips, hits kerb

Experiment 34
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 11375 * 6 = 68250
- Learning rate: 1e-6
- Epoch: **4**
- Training time: 389 s
- Samples per second: 175
- Track 1 performance: Drifts right, drives on double yellow line, drives on red and white rumble strips, drives on to bridge, turns left, hits wall

Experiment 35
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 12994 * 6 = **77964**
- Learning rate: 1e-6
- Epoch: 4
- Training time: 444 s
- Samples per second: 176
- Track 1 performance: Turns right, hits kerb

Experiment 36
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 12994 * 6 = 77964
- Learning rate: 1e-6
- Epoch: **8**
- Training time: 444 s
- Samples per second: 176
- Track 1 performance: Drifts right, drives on double yellow line, stops on right side

Experiment 37
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 14621 * 6 = **87726**
- Learning rate: 1e-6
- Epoch: 8
- Training time: 507 s
- Samples per second: 174
- Track 1 performance: Goes straight, drifts right, drives on red and white rumble strips, turns right, hits kerb

Experiment 38
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 15919 * 6 = **95514**
- Learning rate: 1e-6
- Epoch: 8
- Training time: 555 s
- Samples per second: 172
- Track 1 performance: Goes straight, drifts right, turn right, hit kerb

Experiment 39
- Image: Center, left, right, **flip left**, **flip right**, color, vertical crop, normalized, centered
- Train set size: 15919 * 5 = **79595**
- Learning rate: 1e-6
- Epoch: 8
- Training time: 557 s
- Samples per second: 143
- Track 1 performance: Drift right, hit kerb

Experiment 40
- Image: Center, left, right, **flip**, color, vertical crop, normalized, centered
- Train set size: 17445 * 6 = **104670**
- Learning rate: 1e-6
- Epoch: **16**
- Training time: 617 s
- Samples per second: 171
- Track 1 performance: Go straight, drift right, drive on double yellow lines, drive on red and white rumble strips, go straight, go over kerb

Experiment 41
- Image: Center, left, right, flip, color, vertical crop, normalized, centered
- Train set size: 11016 * 6 = **66096**
- Learning rate: 1e-6
- Epoch: 16
- Training time: 390 s
- Samples per second: 169
- Track 1 performance: Turn left, hit kerb

Experiment 42
- Image: Center, left, right, **flip left**, **flip right**, color, vertical crop, normalized, centered
- Train set size: 17445 * 5 = **87225**
- Learning rate: 1e-6
- Epoch: 16
- Training time: 1222 s
- Samples per second: 71.4
- Track 1 performance: Go straight, drift right, drive close to double yellow lines, go straight after red and white rumble strips, go over kerb

Experiment 43
- Image: Center, left, right, flip left, flip right, color, vertical crop, normalized, centered
- Train set size: 17445 * 5 = 87225
- Learning rate: 1e-6
- Epoch: 16
- Training time: 692 s
- Samples per second: 126
- Track 1 performance: Go straight, drift left, drive close to double yellow lines, go straight after red and white rumble strips, go over kerb

**Experiment 44**
- Image: Center, left, right, **flip**, color, vertical crop, normalized, centered
- Train set size: 17445 * 6 = **104670**
- Learning rate: 1e-6
- Epoch: 16
- Training time: 713 s
- Samples per second: 147
- Track 1 performance: Drift right, go over kerb

Experiment 45
- Image: Center, left, right, **flip left**, **flip right**, color, vertical crop, normalized, centered
- Train set size: 17445 * 5 = **87225**
- Learning rate: 1e-6
- Epoch: **32**
- Training time: 1319 s
- Samples per second: 66.1
- Track 1 performance: Go straight, drift right, hit kerb

Experiment 46
- Image: Center, left, right, flip left, flip right, color, vertical crop, normalized, centered
- Train set size: 17445 * 5 = 87225
- Learning rate: **1e-7**
- Epoch: **16**
- Training time: 661 s
- Samples per second: 132
- Track 1 performance: Turn left, hit kerb

Experiment 47
- Image: Center, left, right, flip left, flip right, color, vertical crop, normalized, centered
- Train set size: 17445 * 5 = 87225
- Learning rate: 1e-7
- Epoch: **32**
- Training time: 1368 s
- Samples per second: 25.5
- Track 1 performance: Go straight, turn left after red and white rumble strips, drift left, hit kerb

__Experiment 48__
- Image: Center, left, right, flip left, flip right, color, vertical crop, normalized, centered
- Train set size: 17445 * 5 = 87225
- Learning rate: **1e-8**
- Epoch: 32
- Training time: 1328 s
- Samples per second: 26.3
- Track 1 performance: Drift right, weave left and right, drove over bridge, drift right, hit kerb

Experiment 49
- Image: Center, left, right, flip left, flip right, grayscale, vertical crop, normalized, centered
- Train set size: 17445 * 5 = 87225
- Learning rate: 1e-8
- Epoch: 32
- Training time: 1111 s
- Samples per second: 31.4
- Track 1 performance: Drift left, drive on double yellow lines, drive center in between red and white rumble strips, drift right after red and white rumble strips, hit kerb

Experiment 50
- Data: 2017-02-16-center-1
- Image: Center, grayscale, vertical crop, normalized, centered
- Train set size: 4518
- Batch size: 32
- Learning rate: 1e-8
- Epoch: 32
- Training time: 149.84 s
- Samples per second: 30.15
- Track 1 performance: Go straight, drift left, drive on double yellow lines, turn right at red and white rumble strips, hit kerb

# reflect
Inverse relationship between learning rate and training time
- Low learning rate increases the probability of finding the local minimum at the expense of training time
- High learning rate reduces training time at the expense of not finding the local minimum

Direct relationship between signal and noise
- Grayscale images reduce noise in color images at the expense of color signals
- Cropped images reduce noise above the horizon at the expense of signals for when going up or down slope

Normal data distribution
- Left and right images reduce driving straight bias

Inverse relationship between batch size and memory usage
- Large batch size improves gradient estimation accuracy at the expense of memory usage
- Small batch size reduces memory usage at the expense of gradient estimation accuracy
- Source: http://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network