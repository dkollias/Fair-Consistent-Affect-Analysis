# Bridging the Gap: Protocol Towards Fair and Consistent Affect Analysis

**The project is an official implementation of our paper:**  

> **[Bridging the Gap: Protocol Towards Fair and Consistent Affect Analysis]() (IEEE FG 2024)**
> 
> **Guanyu Hu, Eleni Papadopoulou, Dimitrios Kollias, Paraskevi Tzouveli, Jie Wei and Xinyu Yang**

The increasing integration of machine learning algorithms in daily life underscores the critical need for fairness and equity in their deployment. As these technologies play a pivotal role in decision-making, addressing biases across diverse subpopulation groups, including age, gender, and race, becomes paramount. Automatic affect analysis, at the intersection of physiology, psychology, and machine learning, has seen significant development. However, existing databases and methodologies lack uniformity, leading to biased evaluations. 
This work addresses these issues by analyzing six affective databases, annotating demographic attributes, and proposing a common protocol for database partitioning. Emphasis is placed on fairness in evaluations. Extensive experiments with baseline and state-of-the-art methods demonstrate the impact of these changes, revealing the inadequacy of prior assessments. The findings underscore the importance of considering demographic attributes in affect analysis research and provide a foundation for more equitable methodologies.


## 1 Access to Developments of our Work

**If you are an academic (i.e., a person with a permanent position at a university, e.g. a professor, but not a Post-Doc or a PhD/PG/UG student), to request the developments of our work (i.e., annotations, partitions, and pre-trained models-checkpoints), please: i) send an email to d.kollias@qmul.ac.uk with subject: Fair and Consistent Affect Analysis request by academic; ii) use your official academic email (as data cannot be released to personal emails); iii) include in the email the [AGREEMENT FORM](), the reason why you require access to these developments, and your official academic website.**

**If you are from industry and you want to acquire the developments of our work (i.e., annotations, partitions, and pre-trained models-checkpoints), please send an email from your official industrial email to d.kollias@qmul.ac.uk with subject: Fair and Consistent Affect Analysis request from industry and explain the reason why the database access is needed; also specify if it is for research or commercial purposes.**


## 2 Datasets

### 2.1 AffectNet 📷 - EXPR & VA

You can download the dataset from http://mohammadmahoor.com/affectnet/

**EXPR-7/8**
- New Partition and Annotations (5 columns):
	- The split between AffectNet 7 and AffectNet 8 is exactly the same. By simply deleting the images and corresponding labels for 'contempt', AffectNet 7 can be derived from AffectNet 8
	- 8 class CSV Path: `AffectNet/annotations/EXPR`
		- Expression Labels: ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
	- Header: [name, expression, age, gender, race]
	- Demographic Labels:
		- Gender: [Female, Male]
		- Age: [00-02, 03-09, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+]
		- Race: [White, Black, Asian, Indian]

**VA**
- New Partition and Annotations:
	- CSV Path: `AffectNet/annotations/VA`
	- Header: [name, valence, arousal, age, gender, race]
	- Demographic Labels:
		- Gender: [Female, Male]
		- Age: [00-02, 03-09, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+]
		- Race: [White, Black, Asian, Indian]

### 2.2 DISFA 🎥 - AU

You can download the dataset from http://mohammadmahoor.com/disfa/

- New Partition and Annotations:
	- 12 AUs
	- CSV Path: `DISFA/annotations`
	- Header: [name, AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26, age, gender, race]
		- AU Labels: [1: present, 0: absent]
	- Demographic Labels:
		- Gender: [Female, Male]
		- Age: [10-19, 20-29, 30-39, 40-49]]
		- Race: [White, Black, Asian, Indian]

### 2.3 EmotioNet 📷 - AU

You can download the dataset from https://cbcsl.ece.ohio-state.edu/dbform_emotionet.html

- New Partition and Annotations:
	- 11 AUs
	- CSV Path: `EmotioNet/annotation`
	- Header: [name, AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU17, AU20, AU25, AU26, age, gender, race]
	- AU: [1: present, 0: absent, -1: unknown]
	- Demographic Labels:
		- Gender: [Female, Male]
		- Age: [00-02, 03-09, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+]
		- Race: [White, Black, Asian, Indian]

### 2.4 GFT 🎥 - AU

You can download the dataset from https://osf.io/7wcyz/

- New Partition and Annotations:
	- 14 AUs
	- Path: `RAF_AU/annotation`
	- Header: [name, AU1, AU2, AU4, AU5, AU6, AU7, AU9, AU10, AU11, AU12, AU15, AU17, AU23, AU24, age, gender, race]
	- Demographic Labels:
		- Gender: [Female, Male]
		- Age: [21, 22, 23, 24, 25, 26, 27, 28]
		- Race: [White, Black, Asian, Other]

### 2.5 RAF-DB 📷 - EXPR

You can download the dataset from http://www.whdeng.cn/raf/model1.html

- New Partition and Annotations:
	- 7 Expressions
	- CSV Path: `RAF_DB/annotation`
	- Header: [name, gender, race, age, expression]
		- Expression Labels: ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
	- Demographic Labels:
		- Gender: [Female, Male, Unsure]
		- Age: [00-03, 04-19, 20-39, 40-69, 70+]
		- Race: [White, Black, Asian, Indian]

### 2.6 RAF-AU 📷 - AU

You can download the dataset from http://whdeng.cn/RAF/model3.html

- New Partition and Annotations:
	- 13 AUs
	- CSV Path: `RAF_AU/annotation`
	- Header: [name, AU1, AU2, AU4, AU5, AU6, AU9, AU10, AU12, AU16, AU17, AU25, AU26, AU27, age, gender, race]
	- Demographic Labels:
		- Gender: [Female, Male]
		- Age: [00-02, 03-09, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+]
		- Race: [White, Black, Asian, Indian]

## 3 Models

Support the following models:

| Model Type   | Models                                                                                                                                       |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| ResNet       | resnet18, resnet34, resnet50, resnet101, resnet152                                                                                           |
| ResNeXt      | resnext50_32x4d, resnext101_32x8d, resnext101_64x4d                                                                                          |
| Swin         | swin_t, swin_s, swin_b, swin_v2_t, swin_v2_s, swin_v2_b                                                                                      |
| VGG          | vgg11, vgg16, vgg19                                                                                                                          |
| ViT          | vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14                                                                                             |
| iResNet      | iresnet101                                                                                                                                   |
| EfficientNet | efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b6, efficientnet_b7, efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l |
| DenseNet     | densenet121, densenet161, densenet201                                                                                                        |
| ConvNeXt     | convnext_tiny, convnext_small, convnext_base, convnext_large                                                                                 |
| POSTER++     | POSTER++                                                                                                                                     |
| DAN          | DAN                                                                                                                                          |
| MT-EffNet    | MT-EffNet                                                                                                                                    |
| MA-Net       | MA-Net                                                                                                                                       |
| EAC          | EAC                                                                                                                                          |
| DACL         | DACL                                                                                                                                         |
| EmoGCN       | EmoGCN                                                                                                                                       |
| CTC          | abaw5_ctc                                                                                                                                    |
| SITU         | abaw5_situ                                                                                                                                   |
| FUXI         | FUXI                                                                                                                                         |
| ME-GraphAU   | ME-GraphAU                                                                                                                                   |
| AUNets       | AUNets                                                                                                                                       |

## 4 Requirements

Our experiments are conducted in the following environments:

- Python 3.11
- PyTorch 2.3
- CUDA 11.8
- Torchvision 0.18

You can set up your environment by following the instructions in `ENVIRONMENT.md`.

## 5 Code Structure

```txt
Fair-Consistent-Affect-Analysis   # Project code root
├── config                        # Configuration and parameters
├── fair                          # Fairness metrics
├── model_utils                   # Model utilities
├── ENVIRONMENT.md                # Requirements
├── run.py                        # Main
├── output                        # Created automatically
└── README.md                     # Project readme
```

## 6 Data Preparation

Your data needs to be structured in the following format:

```text
Dataset_root                   # dataset root
├── AffectNet                  # AffectNet
│   ├── images
│   │   ├── train_set
│   │   │   ├── 0.jpg
│   │   │   └── ...
│   │   └── val_set
│   │       ├── 0.jpg
│   │       └── ...
│   └── annotations
│       ├── EXPR
│       │   ├── train.csv
│       │   ├── valid.csv
│       │   └── test.csv
│       └── VA
│           ├── train.csv
│           ├── valid.csv
│           └── test.csv
├── DISFA                      # DISFA
│   ├── images
│   │   ├── LeftVideoSN001
│   │   │   ├── 00001.jpg
│   │   │   └── ...
│   │   └── RightVideoSN001
│   │       ├── 00001.jpg
│   │       └── ...
│   └── annotations
│       ├── train.csv
│       ├── valid.csv
│       └── test.csv
├── EmotioNet                  # EmotioNet
│   ├── images
│   │   ├── test
│   │   │   ├── 000001.jpg
│   │   │   └── ...
│   │   └── validation
│   │       ├── 000001.jpg
│   │       └── ...
│   └── annotations
│       ├── train.csv
│       ├── valid.csv
│       └── test.csv
├── GFT                        # GFT
│   ├── images
│   │   ├── 001A
│   │   │   ├── 1.jpg
│   │   │   └── ...
│   │   └── 001B
│   │       ├── 1.jpg
│   │       └── ...
│   └── annotations
│       ├── train.csv
│       ├── valid.csv
│       └── test.csv
├── RAF-AU                     # RAF-AU
│   ├── images
│   │   ├── 0001_aligned.jpg
│   │   └── ...
│   └── annotations
│       ├── train.csv
│       ├── valid.csv
│       └── test.csv
└── RAF-DB                     # RAF-DB
    ├── images
    │   ├── test_0001.jpg
    │   ├── ...
    │   ├── train_00001.jpg
    │   └── test.csv
    └── annotations
        ├── train.csv
        ├── valid.csv
        └── test.csv
```

## 7 Config

To train or test, you need to modify the root path in `./config/config.yaml` to your own path.

**Root Path**

```yaml
dataset_root: &dataset_root /path/to/Dataset_root
restructured_csv_root: &restructured_csv_root /path/to/Dataset_root
output_root: /path/to/Fair-Consistent-Affect-Analysis
code_root: /path/to/Fair-Consistent-Affect-Analysis
```

**Dataset Path**

If you structured your dataset repository following the [Data Preparation](#data-preparation) guidelines, you do not need to change any content below. Otherwise, you need to change the paths to match your own directory structure.

- `[Dataset Name]_csv_root`: Dataset CSV folder, where CSV files must follow the `annotations` folder structure as described in the [Data Preparation](#data-preparation) section.
- `[Dataset Name]_num_class`: For expression datasets, it should be the number of expressions. For AU datasets, it should be the number of AUs. For VA datasets, it should be set to 2.

The following is an example for the dataset AffectNet-7:

```yaml
AffectNet-7_dataset_root: !join_path [ *dataset_root, AffectNet/images ]  # Dataset folder
AffectNet-7_csv_root: !join_path [ *restructured_csv_root, AffectNet/annotations/EXPR/expr7 ]  # Dataset CSV folder
AffectNet-7_num_class: 7  # Number of classes
```

## 8 Training

**Training Parameters**

- `dataset`: Controls the training dataset. The following datasets are available:
	- EXPR: AffectNet-7, AffectNet-8, RAF-DB
	- AU: EmotioNet, GFT, RAF-AU, DISFA
	- VA: AffectNet-VA
- `model`: Controls the training model. The models available can be found in the [Models](#models) section.
- `fair`: Run fair validation.  

**Training Examples**

- To run training on the dataset `AffectNet-7` using the model `resnet18`:

```shell
python run.py --dataset AffectNet-7 --model resnet18 -bs 128 --fair
```

- To run training on the dataset `EmotioNet` using the model `convnext_base`:

```shell
python run.py --dataset EmotioNet --model convnext_base -bs 128 --fair
```

**Output**  

The output will be saved to `Fair-Consistent-Affect-Analysis/output`

- `checkpoint`
	- `*.pth`: Checkpoint file
	- `*_fair_results.pkl`: Detailed fairness results
	- `*_inference_results.pkl`: Inference results of the test set
- `tensorboard`: TensorBoard file
- `best_result.txt`: Best result of global validation and its fairness result
- `fair_result`: Fairness results of each epoch
- `result`: Global validation results of each epoch

**output example:**

```text
DISFA-convnext_base-[0524-1725]-lr0.0001-bs128                   
├── checkpoint                  
│   ├── convnext_base_DISFA_best_f1_macro_threshold0.8.pth
│   ├── convnext_base_DISFA_best_f1_macro_threshold0.8_fair_results.pkl
│   └── convnext_base_DISFA_best_f1_macro_threshold0.8_inference_results.pkl
├── tensorboard
│   └── events.out.tfevents.abc.43074.0
├── best_result.txt
├── fair_result.txt
├── loss.txt
└── result.txt
```

## 9 Testing

- `eval`: Activate Evaluation mode.
- `checkpoint_path`: Path to the checkpoint `pth` file

**Training Examples**

- To test the `vit_b_16` checkpoint pre-trained on the dataset `EmotioNet`:

```shell
python run.py --dataset AffectNet-VA -bs 128 --fair --eval --ckeckpoint_path /path/to/vit_b_16_AffectNet-VA.pth
```

- To test the `resnext50` checkpoint pre-trained on the dataset `RAF-DB`:

```shell
python run.py --dataset AffectNet-7 -bs 128 --fair --eval --ckeckpoint_path /path/to/resnext50_32x4d_RAF-DB.pth
```

## 10 Acknowledgements

This repo is based on the following projects, We thank the authors a lot for their valuable efforts.

[POSTER++](https://github.com/Talented-Q/POSTER_V2), [DAN](https://github.com/yaoing/DAN), [HSEmotion](https://github.com/av-savchenko/face-emotion-recognition), [MA-Net](https://github.com/zengqunzhao/MA-Net), [EAC](https://github.com/zyh-uaiaaaa/Erasing-Attention-Consistency), [DACL](https://github.com/amirhfarzaneh/dacl), [Emotion-GCN](https://github.com/PanosAntoniadis/emotion-gcn), [ME-GraphAU](https://github.com/CVI-SZU/ME-GraphAU), [AUNets](https://github.com/BCV-Uniandes/AUNets)

## 11 Citation

Please cite the below paper in your publications if you use our developments (annotations, partitions, and/or trained models). BibTeX reference is as follows:

```bibtex
@article{hu2024bridging,
  title={Bridging the Gap: Protocol Towards Fair and Consistent Affect Analysis},
  author={Hu, Guanyu and Papadopoulou, Eleni and Kollias, Dimitrios and Tzouveli, Paraskevi and Wei, Jie and Yang, Xinyu},
  journal={arXiv preprint arXiv:2405.06841},
  year={2024}
}
```
