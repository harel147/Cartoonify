# 046211 - Cartoonify

## Agenda
* Project overview
* Prerequisits
* Building the Dataset
* Training & Evaluation
* Credits and References


### Project overview

Data augmentation is a widely recognized approach utilized to enhance the accuracy of neural networks across various tasks. By applying data augmentation techniques, models are able to generalize better and exhibit improved performance when faced with unseen data. Particularly, Generative Adversarial Networks (GANs) have garnered substantial attention in the realm of image generation. Researchers have previously delved into the application of GANs for data augmentation, revealing their immense potential in augmenting model performance. Through the utilization of GANs, models can benefit from the synthesized data, leading to enhanced generalization and ultimately improved accuracy across multiple tasks.

Our objective is to investigate the efficacy of "cartoon augmentation" using Generative Adversarial Networks (GANs) in the context of image classification. As far as we are aware, we are the pioneers in exploring the concept of "cartoon augmentation." To assess the applicability and potential advantages of this technique, we have specifically opted for facial expression classification as a case study. By focusing on this particular task, we aim to thoroughly examine the implementation of "cartoon augmentation" and ascertain its potential benefits in enhancing the accuracy and performance of image classification models.

To create cartoon versions of the original dataset, we used [CartoonGAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf):

![image](https://github.com/harel147/Cartoonify/assets/63463677/666d18fc-2b59-4de0-97f8-02273468fb11)

In our study, we chose ResNet18 as our model, which has been utilized in prior research. Previous studie reached an accuracy of 73% with an optimized ResNet18, which closely approaches the highest reported accuracy of 77% found in the literature. 

### Environment Setting
You can install all necessary packages by running the following command:
```bash
pip install -r requirements.txt
```

### Building the Dataset
Our study focused on FER2013 dataset:
* Train set size 28k
* Validation set size 3.5k
* Test set size 3.5k
  
FER2013 samples Vs augmented version of the same samples:
![image](https://github.com/harel147/Cartoonify/assets/63463677/5f58555b-d45c-49cb-bde8-fb98b4e06e7d)

You can create a cartoon augmented version of the dataset, by running the following command:
```bash
python generate_cartoon_from_folder.py \
    --style Paprika \
    --input_dir ./CartoonGAN_for_torch/test_img/ \
    --output_dir ./CartoonGAN_for_torch/test_ouput/ \
    --gpu 0
```
### Training & Evaluation
During the training process, we implemented various augmentation modes:
* Implemented random augmentation based on a selected percentage
* Trained the models exclusively on augmented data
* Combined the augmented dataset with the original dataset

During the evaluation process, we implemented various modes:
* Test on the original test set
* Evaluated the models solely on the augmented test set
* Utilized two models simultaneously, exploring techniques such as ensemble methods (sum or max on the output)

You can train the model using `train_facial_expression.py`, choosing different hyperparameters and training modes:
```bash
python train_facial_expression.py \
    --epochs {default=300} \
    --batch_size {default=64} \
    --optimizer {adam (default), sgd} \
    --scheduler {reduce (default), cos} \
    --lr_sgd {default=0.1} \
    --lr_adam {default=0.001} \
    --momentum {default=0.9} \
    --weight_decay {default=1e-4} \ 
    --data_path {default='./FER2013'} \ 
    --cartoon_prec {default=0.5} \
    --test_mode {regular (default), cartoon} \
    --train_on_united {no (default), yes} \
    --weights_init {imagenet (default), xavier}
    
```
To change the cartoon augmentations precentage set `--cartoon_prec` to your desired precentage.

To run augmentations at test time, run with `--test_mode cartoon`.

To train on a dataset combined of the original train set images + all the augmented train set images, run with `--train_on_united yes`.




### Models results
To examine the effect of Contrastive-center loss regularization, we ran the following experiments in addition to the baseline: 
* Contrastive-center loss regularization with Adagrad optimizer with lr=0.001 and 𝜆=1
* Contrastive-center loss regularization with Adam optimizer with lr=0.002 and 𝜆=0.55

As seen below, the regularization did not effect the train loss.
![image](https://user-images.githubusercontent.com/74931703/214831357-8bbc8245-6f6a-432a-8244-9f56d29cb2a4.png)

In addition, in all cases on our validation set, we converge relatively to the same value but in a different pace.
![image](https://user-images.githubusercontent.com/74931703/214831623-e4685ab0-2fb0-4726-8538-40a76ac8f7fc.png)

Note that the value of the Contrastive-center loss is small from the first iteration - approximal $2\cdot10^(-4)$ as opposed to the Cross-Entropy loss that starts at the value of about 4. Due to the extreme size difference between the CCL regularization and the Cross Entropy loss, the CCL regularization has a minor impact on the train and validation coverage as seen inthe graphs above.

| Model | Top-1 (%) | Top-5 (%) |
| ------------- | ------------- | ------------- |
| Baseline | **79.09** | 92.78 |
| Contrastive-center loss with Adagrad | 78.98 | **93.24** |
| Contrastive-center loss with Adam | 78.52 | 92.82 |

As seen in the table above, all 3 experiments obtain similar top-1 accuracy, whereas the Contrastive-center loss experiment with Adagrad optimizer increased the top-5 accuracy by 0.5%, showing that the regularization improved the generalization of the model.

### Prerequisits
| Library  | Version |
| ------------- | ------------- |
| `Python`  | `3.8.16`  |
| `torch`  | `1.13.0`  |
| `numpy`  | `1.21.6`  |
| `torchaudio`  | `0.13.0`  |
| `torchvision`  | `0.14.0`  |
| `pandas`  | `1.3.5`  |
| `librosa`  | `0.8.1`  |
| `matplotlib`  | `3.2.2`  |

### Running our model
1. **Download VoxCeleb1 dataset** <br>
   Run <br>
   `python arrange_dataset.py [--download] --n_speakers <num_of_speakers> --dataset_dir <path_to_dataset> --checkpoint_dir <path>` 

  | Argument  | Explanation |
  | ------------- | ------------- |
  | `n_speakers`  | the number of speakers wanted in the dataset, needs to be <= 1251 |
  | `download`  | if Added to commandline, then downloading to `dataset_dir` the VoxCeleb1 dataset |
  | `dataset_dir`  | path to the directory of the dataset |
  | `resplit`  | if Added to commandline, then re-splitting the dataset to train, validation and loss accordint to `train_size` and `val_size` |
  | `train_size`  | train's element of the dataset, by default 0.6 |
  | `val_size`  | validation's element of the dataset, by default 0.2 |
  
2. **Train our model** <br>
  To train our model, run <br>
  `python train_model.py --ccl_reg --n_speakers <num_of_speakers> --dataset_dir <path_to_dataset> --checkpoint_dir <path>`

  | Argument  | Explanation |
  | ------------- | ------------- |
  | `n_speakers`  | the number of speakers in the dataset, needs to be <= 1251 |
  | `dataset_dir`  | path to the directory of the dataset |
  | `checkpoint_dir`  | path to save the checkpoints |
  | `ccl_reg`  | if Added to commandline, then training with contrastive-center loss regularization |
  | `batch_size`  | train's batch_size, by default 64 |
  | `n_epochs`  | number of epochs, by default 20 | 
  <br>
    Note: There are more arguments, such as Renet's learning rate. To see all of them run: <code>python train_model.py -h</code>

### References
We based our project on the results of the following papers and github repositories:
<br>[1] S. Bianco, E. Cereda and P. Napoletano, "Discriminative Deep Audio Feature Embedding for Speaker Recognition in the Wild," 2018 IEEE 8th International Conference on Consumer Electronics - Berlin (ICCE-Berlin), Berlin, Germany, 2018, pp. 1-5, doi: 10.1109/ICCE-Berlin.2018.8576237.
<br>[2] M. Jakubec, E. Lieskovska and R. Jarina, "Speaker Recognition with ResNet and VGG Networks," 2021 31st International Conference Radioelektronika (RADIOELEKTRONIKA), Brno, Czech Republic, 2021, pp. 1-5, doi: 10.1109/RADIOELEKTRONIKA52220.2021.9420202.
<br>[3] https://github.com/samtwl/Deep-Learning-Contrastive-Center-Loss-Transfer-Learning-Food-Classification-/tree/master
