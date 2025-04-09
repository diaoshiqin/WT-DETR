# WT-DETR: Wavelet-Enhanced DETR for Robust Tiny Object Detection via Multi-Scale Feature Optimization

## 📰 Abstract

Tiny object detection remains a challenging task in computer vision, with applications spanning remote sensing, intelligent transportation, and aerial surveillance. Despite advancements in DETR-based methods, detecting tiny objects is hindered by limited receptive fields, aliasing artifacts, and the loss of fine-grained details.
To overcome these limitations, we propose **WT-DETR**, a novel framework that leverages wavelet transform-based optimizations. WT-DETR introduces **Wave Field Convolution (WFC)** to expand the receptive field while capturing global context and structural features with minimal parameter overhead. Furthermore, **Wavelet Anti-Aliasing Downsampling (WTD)** replaces conventional downsampling, reducing aliasing and preserving crucial details.
By integrating these components, WT-DETR effectively enhances multi-scale feature preservation while maintaining computational efficiency. Experiments on the **VisDrone2019** and **SIMD** datasets demonstrate WT-DETR's superior performance, achieving *mAP₅₀* scores of **59.65%** and **81.0%**, respectively, surpassing state-of-the-art methods by significant margins. These results establish WT-DETR as a new benchmark for tiny object detection, offering a promising solution for precise detection in complex environments.

---

## ✨ Highlights

### 🆕 WT-DETR  
We propose **WT-DETR**, a novel framework tailored for tiny object detection, which integrates global contextual understanding and fine-grained detail preservation through wavelet transform-based modules. This design addresses the core limitations of DETR variants when dealing with small-scale targets in complex scenes.

### 🆕 WFC and WTD  
WT-DETR features **Wave Field Convolution (WFC)** and **Wavelet Anti-Aliasing Downsampling (WTD)**. WFC expands the receptive field and captures global structural context with minimal parameters, while WTD mitigates aliasing and preserves critical small-object features during downsampling, significantly enhancing multi-scale representation quality.

### 🆕 Experiment Result  
Extensive experiments on **VisDrone2019** and **SIMD** datasets demonstrate that WT-DETR delivers superior detection performance and faster convergence. It achieves *mAP₅₀* scores of **59.65%** and **81.0%**, respectively, outperforming state-of-the-art methods and setting a new benchmark in tiny object detection.

---



## Flowchart

![WT-DETR Flowchart](WT-DETR-main/images/WT-DETR.png)
Overall framework of **WT-DETR**. The WT-DETR model introduces innovations in three key modules. The **WTHE** module integrates the WFC and WTD submodules through recursive and interactive fusion mechanisms. The **WFC** module decomposes images via wavelet transform and applies convolutions on low-frequency features to expand the receptive field and enhance both global and detailed feature capture capabilities. The **WTD** module utilizes wavelet anti-aliasing downsampling to reduce resolution loss and further improve the detection accuracy of small objects.



## ⚙️ Installation

### 🔧 Create environment and install dependencies

1. Create a conda virtual environment and activate it.
    ```shell
    conda create -n wtdetr python=3.8 -y
    conda activate wtdetr
    pip install -r requirements.txt
    ```                        


2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```shell
    conda install pytorch torchvision -c pytorch
    ```

    Note: Make sure that your compilation CUDA version and runtime CUDA version match.
    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

    If you build PyTorch from source instead of installing the prebuilt pacakge,
    you can use more CUDA versions such as 11.7.

3. Install mmcv-full, we recommend you to install the pre-build package as below.

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    ```

    Please replace `{cu_version}` and `{torch_version}` in the url to your desired one. For example, to install the latest `mmcv-full` with `CUDA 11。7` and `PyTorch 1.13.1`, use the following command:

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.1/index.html
    ```

    See [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions.
    Optionally you can choose to compile mmcv from source by the following command

    ```shell
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
    cd ..
    ```

    Or directly run

    ```shell
    pip install mmcv-full
    ```


- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

The compatible MMDetection and MMCV versions are as below. Please install the correct version of MMCV to avoid installation issues.

| MMDetection version |    MMCV version     |
|:-------------------:|:-------------------:|
| master              | mmcv-full>=1.3.3, <1.4.0 |
| 2.12.0              | mmcv-full>=1.3.3, <1.4.0 |
| 2.11.0              | mmcv-full>=1.2.4, <1.4.0 |
| 2.10.0              | mmcv-full>=1.2.4, <1.4.0 |
| 2.9.0               | mmcv-full>=1.2.4, <1.4.0 |
| 2.8.0               | mmcv-full>=1.2.4, <1.4.0 |
| 2.7.0               | mmcv-full>=1.1.5, <1.4.0 |
| 2.6.0               | mmcv-full>=1.1.5, <1.4.0 |
| 2.5.0               | mmcv-full>=1.1.5, <1.4.0 |
| 2.4.0               | mmcv-full>=1.1.1, <1.4.0 |
| 2.3.0               | mmcv-full==1.0.5    |
| 2.3.0rc0            | mmcv-full>=1.0.2    |
| 2.2.1               | mmcv==0.6.2         |
| 2.2.0               | mmcv==0.6.2         |
| 2.1.0               | mmcv>=0.5.9, <=0.6.1|
| 2.0.0               | mmcv>=0.5.1, <=0.5.8|


Note: You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.


## 📂 Dataset Structure

1.The folder structure of the VisDrone2019 dataset used in this project is organized as follows:

Please download [VisDrone 2019](https://github.com/VisDrone/VisDrone-Dataset) dataset and organize them as following:
```
datasets/
└── visdrone2019/
    ├── images/
    ├── labels/
    ├── train/
    ├── val/
    └── test/
```

2.The folder structure of the Simd dataset used in this project is organized as follows:

Please download [Simd](https://github.com/ihians/simd/tree/master) dataset and organize them as following:
```
datasets/
└── Simd/
    ├── images/
    ├── labels/
    ├── train/
    ├── val/
    └── test/
```


## Acknowledgements

This research was supported by the following funding sources:  
- The Research Programs of the Henan Provincial Department of Science and Technology (252102210034, 232102210054)  
- The Chongqing Natural Science Foundation (CSTB2023NSCQ-MSX0070)  
- The Henan Province Key Research and Development Project (231111212000)  
- The Aviation Science Foundation (20230001055002)  
- The Henan Center for Outstanding Overseas Scientists (GZS2022011)


## 📖 Citation (Under Review)

If you find **WT-DETR** useful in your research, please consider citing our work:

> **"WT-DETR: Wavelet Transform Enhanced DETR for Tiny Object Detection"**  
> Xiaoyan Shao, Shiqin Diao, Lingling Li, Xuezhuan Zhao, Yang Mei, Zonghao Zhu  
> *Submitted to The Visual Computer (Springer), 2025.*

We will update the citation with the official DOI once the paper is accepted.













