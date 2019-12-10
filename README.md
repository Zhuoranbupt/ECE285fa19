ECE 285 MLIP Project C Multi-Object detection using Deep Learning
===========================
This is a simplified introduction of ECE 285 project--Multi-Object detection.

****
	
|Author|
|---
|Fei Xue|
|Bo Zhou|
|Zhuoran Liu|

## Invironment configuration
---

```

git clone https://github.com/Zhuoranbupt/ECE285fa19/tree/master
pip install --user scipy == 1.3.0
pip install --uer scikit-image

```

## Dataset
---
The dataset we used in the project is PascalVOC2012, the JPEGImages for images and Annotations for labels.

```
ln -s /datasets/ee285f-public/PascalVOC2012 
```

## Code organization
---
dataloder.py -- imports the data from the file and outputs image-label pairs


## Train
---

## Test
---

## Acknowledgement
---
### YOLOv3: An Incremental Improvement
```
@article{DBLP:journals/corr/abs-1804-02767,
  author    = {Joseph Redmon and
               Ali Farhadi},
  title     = {YOLOv3: An Incremental Improvement},
  journal   = {CoRR},
  volume    = {abs/1804.02767},
  year      = {2018},
  url       = {http://arxiv.org/abs/1804.02767},
  archivePrefix = {arXiv},
  eprint    = {1804.02767},
  timestamp = {Mon, 13 Aug 2018 16:48:24 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1804-02767},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


Redmon J, Farhadi A. Yolov3: An incremental improvement[J]. arXiv preprint arXiv:1804.02767, 2018.
https://pjreddie.com/media/files/papers/YOLOv3.pdf

