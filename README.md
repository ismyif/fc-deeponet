# Abstract

Seismic travel time plays a fundamental role in a wide array of geophysical applications. Traditionally, numerical simulation of travel time involves solving the eikonal equation. However, conventional methods are typically limited to simulating the travel-time field for a single source and velocity model at a time. This limitation poses challenges, particularly when dealing with inverse problems that necessitate multiple forward simulations to infer velocity models based on travel-time data excited by different sources. In recent years, machine learning has proven its effectiveness in tackling problems associated with partial differential equations (PDEs). Among these methods, the Deep Operator Network (DeepONet) has gained attention for its adaptable structure and minimal generalization error. In response to the challenges posed by solving the eikonal equation in heterogeneous media, we introduce a modified architecture known as the Fully Convolutional DeepONet (FC-DeepONet). This approach leverages convolutional operations to extract features directly from 2D data and avoid flattening operations that could lead to the loss of important spatial information. The FC-DeepONet model takes the velocity model and source location as input and generates the corresponding travel-time fields as output. Through numerical experiments, we validate the efficacy of our proposed method in accurately predicting travel-time fields induced by sources located at various positions across diverse velocity models. Besides, our approach demonstrates robustness by providing reasonably accurate predictions even in scenarios involving velocity models with irregular topography. This adaptability holds significant promise for practical applications, particularly in cases characterized by complex geological features.

![fc-don](fc_don.png)

```bibtex
@ARTICLE{mei2024fully,
  author={Mei, Yifan and Zhang, Yijie and Zhu, Xueyu and Gou, Rongxi and Gao, Jinghuai},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Fully Convolutional Network-Enhanced DeepONet-Based Surrogate of Predicting the Travel-Time Fields}, 
  year={2024},
  volume={62},
  pages={1-12},
  doi={10.1109/TGRS.2024.3401196}}
```

# Datasets
- [OpenFWI](https://openfwi-lanl.github.io/)

# Requirements
Tested on `python==3.8`. Install the requirements using the following command:
```bash
pip install -r requirements.txt
```
