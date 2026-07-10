# KlipAlign LightGlue ⚡️
## Local Feature Matching at Light Speed

[[`Paper`](https://arxiv.org/abs/2306.13643)]

This repository hosts the inference code of LightGlue, a lightweight feature matcher with high accuracy and blazing fast inference. It takes as input a set of keypoints and descriptors for each image and returns the indices of corresponding points. The architecture is based on adaptive pruning techniques, in both network width and depth.

We release pretrained weights of LightGlue with [SuperPoint](https://arxiv.org/abs/1712.07629), [DISK](https://arxiv.org/abs/2006.13566), [ALIKED](https://arxiv.org/abs/2304.03608) and [SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) local features.

## Installation

```bash
git clone https://github.com/Ayush-pani/klipalign_LightGlue.git && cd klipalign_LightGlue
python -m pip install -e .
```

## Usage

Minimal script to match two images:

```python
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd

# SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
matcher = LightGlue(features='superpoint').eval().cuda()

# or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
extractor = DISK(max_num_keypoints=2048).eval().cuda()
matcher = LightGlue(features='disk').eval().cuda()

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
image0 = load_image('path/to/image_0.jpg').cuda()
image1 = load_image('path/to/image_1.jpg').cuda()

# extract local features
feats0 = extractor.extract(image0)
feats1 = extractor.extract(image1)

# match the features
matches01 = matcher({'image0': feats0, 'image1': feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
matches = matches01['matches']
points0 = feats0['keypoints'][matches[..., 0]]
points1 = feats1['keypoints'][matches[..., 1]]
```

Convenience method:

```python
from lightglue import match_pair
feats0, feats1, matches01 = match_pair(extractor, matcher, image0, image1)
```

## Advanced configuration

- ```n_layers```: Number of stacked self+cross attention layers. Default: 9.
- ```flash```: Enable FlashAttention. Default: True.
- ```mp```: Enable mixed precision inference. Default: False.
- ```depth_confidence```: Controls the early stopping. Default: 0.95, disable with -1.
- ```width_confidence```: Controls the iterative point pruning. Default: 0.99, disable with -1.
- ```filter_threshold```: Match confidence. Default: 0.1

The default values give a good trade-off between speed and accuracy. To maximize accuracy:

```python
extractor = SuperPoint(max_num_keypoints=None)
matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1)
```

To increase speed:

```python
extractor = SuperPoint(max_num_keypoints=1024)
matcher = LightGlue(features='superpoint', depth_confidence=0.9, width_confidence=0.95)
```

Maximum speed is obtained with FlashAttention (automatically used when `torch >= 2.0`) and PyTorch compilation:

```python
matcher = matcher.eval().cuda()
matcher.compile(mode='reduce-overhead')
```

## Benchmark

LightGlue runs at 150 FPS @ 1024 keypoints and 50 FPS @ 4096 keypoints per image on GPU (RTX 3080).

Obtain benchmark plots for your setup:

```
python benchmark.py [--device cuda] [--add_superglue] [--num_keypoints 512 1024 2048 4096] [--compile]
```

## BibTeX citation

```txt
@inproceedings{lindenberger2023lightglue,
  author    = {Philipp Lindenberger and
               Paul-Edouard Sarlin and
               Marc Pollefeys},
  title     = {{LightGlue: Local Feature Matching at Light Speed}},
  booktitle = {ICCV},
  year      = {2023}
}
```

## License

The code and pretrained weights are released under the [Apache-2.0 license](./LICENSE).
