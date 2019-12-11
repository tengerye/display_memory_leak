It is a temporary repository to display the memory leak in pytorch object detection.

The repository is almost the same from the tutorial for object detection (https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).

Requirements are:
Python >=3.6.8
pytorch >=1.4
torchvision >= 0.4

In order to run the script, you need to run `tv-training-code.py`.
The exepected output is:
```
Count of tensors = 210.
Size of cache=1
Epoch: [0]  [ 0/60]  eta: 0:16:05  lr: 0.000090  loss: 3.4009 (3.4009)  loss_classifier: 0.5852 (0.5852)  loss_box_reg: 0.1498 (0.1498)  loss_mask: 2.6170 (2.6170)  loss_objectness: 0.0448 (0.0448)  loss_rpn_box_reg: 0.0042 (0.0042)  time: 16.0838  data: 0.1369  max mem: 0
Count of tensors = 215.
Size of cache=2
Count of tensors = 220.
Size of cache=3
Count of tensors = 220.
Size of cache=3
Count of tensors = 225.
Size of cache=4
Count of tensors = 230.
Size of cache=5
Count of tensors = 235.
Size of cache=6
Count of tensors = 240.
Size of cache=7
Count of tensors = 245.
Size of cache=8
Count of tensors = 250.
Size of cache=9
Count of tensors = 255.
Size of cache=10
Epoch: [0]  [10/60]  eta: 0:12:25  lr: 0.000936  loss: 1.5595 (2.0232)  loss_classifier: 0.3378 (0.3822)  loss_box_reg: 0.1782 (0.1815)  loss_mask: 1.0956 (1.4184)  loss_objectness: 0.0220 (0.0318)  loss_rpn_box_reg: 0.0083 (0.0094)  time: 14.9020  data: 0.0189  max mem: 0
Count of tensors = 260.
Size of cache=11
Count of tensors = 260.
Size of cache=11
Count of tensors = 260.
Size of cache=11
Count of tensors = 260.
Size of cache=11
Count of tensors = 260.
Size of cache=11
Count of tensors = 265.
Size of cache=12
```

Observe that:
1. The number of tensors NOT released increases every iteration;
2. The number of RPN's `_cache` increases every iteration.

In other words, memory leaks.
