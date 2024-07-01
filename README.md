 # AppleLite-ShuffleNet

 ## **Introduction**
- As the apple industry grows, external quality grading has become essential for improving quality and efficiency. It upholds quality standards, assesses and sorts apples by their quality, identifies production issues, and suggests improvements, enabling consumers to choose products according to their preferences. This is crucial for producers, exporters, and importers. Manual grading is time-consuming and labor-intensive, prompting the need for automated sorting through machine vision techniques. In practical scenarios, the grading system should be easily adaptable to mobile devices, embedded systems, and IoT technology, allowing growers to evaluate apple quality anytime, anywhere. Thus, vision grading models must be lightweight, accurate, and fast for effective operation.
**********
 ## **Table of Contents**

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

 ## **Features**
- **Quality Standards Maintenance**: Ensures that apple quality standards are upheld.
- **Assessment and Sorting**: Evaluates and sorts apples based on their quality.
- **Production Issues Identification**: Identifies production issues and suggests improvements.
- **Consumer Preferences**: Enables consumers to select products according to their preferences.
- **Automation Necessity**: Highlights the need for automated sorting due to the time-consuming and labor-intensive nature of manual grading.
- **Machine Vision Techniques**: Uses machine vision techniques for automated sorting.
- **Adaptability**: Grading system is adaptable to mobile devices, embedded systems, and IoT technology.
- **Anytime, Anywhere Evaluation**: Allows growers to evaluate apple quality at any time and place.
- **Model Requirements**: Vision grading models must be lightweight, accurate, and fast for effective operation.

 ## **Installation**

- The operating system used is Windows 10 based on the Pytorch deep learning framework. the hardware configuration is AMD Ryzen 7 4800H, Radeon Graphics @2.9GHz, 16GB DDR4 RAM, Pytorch version 2.1.2, and torchvision version 0.16.2.

 ## **Usage**

- Here's a quick example of how to use Mini-ShufflenetV1 for apple external quality grading:

1. **Load the model**:
```python
model_cfg = dict(
    backbone=dict(type='Mini-ShuffleNetV1'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='StackedLinearClsHead',
        num_classes=5,
        in_channels=360,
    )
)
```

2. **dataloader pipeline**:
```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandomErasing',
        erase_prob=0.2,
        mode='const',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
```

 ## **Dataset**

- To train and evaluate the model, you'll need a dataset

 **Clone the Dataset**:
    
    https://github.com/AIYAU/AppleLite-ShuffleNet/datasets/

 ## **Training**

- To train the Mini-ShufflenetV1 model, use the following command:

```python
data_cfg = dict(
    batch_size = 64,
    num_workers = 4,
    train = dict(
        pretrained_flag = True,
        pretrained_weights = 'datas/Mini-shufflenet_v1_batch1024_imagenet_20200804-5d6cec73.pth',
        freeze_flag = False,
        freeze_layers = ('backbone',),
        epoches = 100,
        )
)
```
## **Evaluation**
- To evaluate the Mini-ShufflenetV1 model, use the following command
```python 
test=dict(
        ckpt = 'logs\Mini-ShuffleNetV1/2024-01-30-20-04-36/Train_Epoch098-Loss0.018.pth',
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
        metric_options = dict(
            topk = (1,5),
            thrs = None,
            average_mode='none'
        )
)
```
 ## Contributions

1. **Development of Mini-ShuffleNetV1**: Created a lightweight model for accurate recognition of Apple's external quality on embedded systems.

2. **Efficiency Improvements**: Achieved a 49.45% reduction in model size, decreased parameters to 0.46M, and reduced computational complexity by 53.33%, maintaining high accuracy and performance.

3. **Enhanced Performance**: Outperformed original ShuffleNetV1 and state-of-the-art ShuffleNetV2 in speed and accuracy, meeting the operational needs of apple farmers.

4. **Practical Applicability**: Demonstrated the model's viability for deployment on mobile devices, providing valuable insights for future research and application.

5. **Addressing Limitations**: Identified potential issues such as unclear images, subjective evaluations, and the need for additional datasets to enhance robustness.

 ## License

- This study and the Mini-ShuffleNetV1 model are released under the MIT License.

 ### MIT License

- Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
- The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
