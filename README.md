# 046211 - Cartoonify - Cartoon Augmentations for Image Classification

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

In our study, we chose ResNet18 as our model, which has been utilized in prior research. Previous studies reached an accuracy of 73% with an optimized ResNet18, which closely approaches the highest reported accuracy of 77% found in the literature. 

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
![image](https://github.com/harel147/Cartoonify/assets/63463677/c20943dc-a8b9-45bf-9425-d8c7eab2ed21)


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
* Random augmentation based on a selected percentage
* Trained the models exclusively on augmented data
* Combined the augmented dataset with the original dataset

During the evaluation process, we implemented various modes:
* Test on the original test set
* Evaluate the models solely on the augmented test set
* Utilize two models simultaneously, exploring techniques such as ensemble methods (sum or max on the output)

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

After training, the complete evaluation of the model will be automatically saved to `./results/<train_folder>/`

##### Evaluation with two models
For utilizing two models simultaneously at evaluation time, one for the original test set and the second for the cartoon test set, use `test_on_2_models_simultaneously.py`:
```
python test_on_2_models_simultaneously.py \
    --test_name {name for the experiment}
    --model_checkpoint_original_testset {path for the first model checkpoints}
    --model_checkpoint_cartoon_testset {path for the second model checkpoints}
```

### Models results
We conducted a total of 39 experiments, and for each experiment, we recorded the following data:
* The best validation accuracy weights
* The graph of the train and validation loss
* The confusion matrix
  
To ensure comprehensive documentation, we organized and recorded all experiment details in a table, allowing for easy reference and analysis of the results:

| trail number	|optimizer|	lr	|batch	|cartoon train	|cartoon test	|weights init	|result|	comments|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|	1	|	adam	|	0.001	|	64	|	0	|	0	|	imagenet	|	60.77	|		|
|	2	|	adam	|	0.001	|	64	|	0.2	|	0	|	imagenet	|	60.02	|		|
|	3	|	adam	|	0.001	|	64	|	0.4	|	0	|	imagenet	|	59.96	|		|
|	4	|	adam	|	0.001	|	64	|	0.6	|	0	|	imagenet	|	60.55	|		|
|	5	|	adam	|	0.0001	|	64	|	0	|	0	|	imagenet	|	63.89	|		|
|	6	|	adam	|	0.0001	|	64	|	0.2	|	0	|	imagenet	|	63.78	|		|
|	7	|	adam	|	0.0001	|	64	|	0.4	|	0	|	imagenet	|	63.19	|		|
|	8	|	adam	|	0.0001	|	64	|	0.6	|	0	|	imagenet	|	60.94	|		|
|	9	|	sgd	|	0.1	|	64	|	0	|	0	|	imagenet	|	58.48	|	add schduler	|
|	10	|	sgd	|	0.1	|	64	|	0.2	|	0	|	imagenet	|	58.04	|	add schduler	|
|	11	|	sgd	|	0.1	|	64	|	0.4	|	0	|	imagenet	|	57.62	|	add schduler	|
|	12	|	sgd	|	0.1	|	64	|	0.6	|	0	|	imagenet	|	58.23	|	add schduler	|
|	13	|	adam	|	0.0001	|	32	|	0	|	0	|	imagenet	|	64.06	|		|
|	14	|	adam	|	0.0001	|	128	|	0	|	0	|	imagenet	|	63.5	|		|
|	15	|	adam	|	0.0001	|	64	|	1	|	1	|	imagenet	|	54.28	|		|
|	16	|	adam	|	0.0001	|	64	|	0.8	|	1	|	imagenet	|	56.73	|		|
|	17	|	adam	|	0.0001	|	64	|	0.6	|	1	|	imagenet	|	56.23	|		|
|	18	|	adam	|	0.0001	|	64	|	0.4	|	1	|	imagenet	|	56.76	|		|
|	19	|	adam	|	0.0001	|	64	|	0.2	|	1	|	imagenet	|	56.34	|		|
|	20	|	adam	|	0.0001	|	64	|	0	|	1	|	imagenet	|	31.99	|		|
|	21	|	sgd	|	0.1	|	32	|	0	|	1	|	imagenet	|	58.93	|	add schduler	|
|	22	|	sgd	|	0.1	|	128	|	0	|	1	|	imagenet	|	54.92	|	add schduler	|
|	23	|	sgd	|	0.1	|	32	|	0.2	|	1	|	imagenet	|	60.3	|	add schduler	|
|	24	|	sgd	|	0.1	|	128	|	0.2	|	1	|	imagenet	|	57.34	|	add schduler	|
|	25	|	adam	|	0.0001	|	32	|	0.5	|	0	|	imagenet	|	63.05	|	train on united train set	|
|	26	|	adam	|	0.0001	|	64	|	0.5	|	0	|	imagenet	|	62.55	|	train on united train set	|
|	27	|	adam	|	0.0001	|	128	|	0.5	|	0	|	imagenet	|	62.64	|	train on united train set	|
|	28	|	sgd	|	0.1	|	32	|	0.5	|	0	|	imagenet	|	57.73	|	train on united train set	|
|	29	|	sgd	|	0.1	|	64	|	0.5	|	0	|	imagenet	|	58.68	|	train on united train set	|
|	30	|	sgd	|	0.1	|	128	|	0.5	|	0	|	imagenet	|	57.82	|	train on united train set	|
|	31	|	adam	|	0.0001	|	32	|	0.1	|	0	|	imagenet	|	63.42	|		|
|	32	|	adam	|	0.0001	|	32	|	0.05	|	0	|	imagenet	|	64.36	|		|
|	33	|	adam	|	0.0001	|	32	|	0.02	|	0	|	imagenet	|	62.75	|		|
|	34	|	adam	|	0.0001	|	32	|	0	|	0	|	xavier	|	56.9	|		|
|	35	|	adam	|	0.0001	|	32	|	0.1	|	0	|	xavier	|	56.56	|		|
|	36	|	adam	|	0.0001	|	32	|	0.2	|	0	|	xavier	|	55.89	|		|
|	37	|	adam	|	0.0001	|	32	|	0.3	|	0	|	xavier	|	56.59	|		|
|	38	|	adam	|	0.0001	|	32	|	1	|	1	|	xavier	|	48.31	|		|
|	39	|	adam	|	0.0001	|	32	|	0.5	|	0	|	xavier	|	57.17	|	train on united train set	|
  <br>


#### Example for graph of the train and validation loss:
![image](https://github.com/harel147/Cartoonify/assets/63463677/a9ea5bdd-7c3a-4e72-aabe-1d34092661bd)
#### Example for graph of validation accuracy:
![image](https://github.com/harel147/Cartoonify/assets/63463677/dd3611de-369f-4e6d-915b-f3083af8b193)


#### Example for the confusion matrix:
![image](https://github.com/harel147/Cartoonify/assets/63463677/81ce961f-aa5b-43eb-ae38-afebfdd5af7b)

### Results Summery
* We observed improved results when using a pre-trained ImageNet model compared to the Xavier initialization method 
* The Adam optimizer outperformed the SGD optimizer
* Utilizing the combined dataset, which included both original and augmented data, yielded very similar results to using only the original dataset
* We noticed a decrease in accuracy when the augmentation percentage exceeded 10%
* It appeared that the task of facial expression classification became more challenging when using only the augmented dataset
* Simultaneously using two models with either a sum or max operation on the output scores vector resulted in decreased accuracy compared to the standard single-model approach

### Conclusion
* "Cartoon augmentation" improves model accuracy when used on a small percentage of the training dataset.
* Extensive use of "cartoon augmentation" in facial expression classification can decrease accuracy due to significant differences between "cartoon faces" and real face images.
* Exploring "cartoon augmentation" in other classification tasks may yield positive results, as different domains can benefit from the stylized features.
* Future research should focus on using less aggressive GANs for "cartoon augmentation" to achieve better outcomes.

### References
We based our project on the results of the following papers and github repositories:
* [CartoonGAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf): Generative Adversarial Networks for Photo Cartoonization (Yang Chen, et al. 2018) 
* [Facial Expression Recognition] (https://github.com/mansikataria/FacialExpressionRecognition)
* [FER2013 - Facial Emotion Recognition] (https://github.com/LetheSec/Fer2013-Facial-Emotion-Recognition-Pytorch/tree/4101684616b1c9bde358fc6a2082e3b9ef121a9c)
* [GAN Augmentation] (https://arxiv.org/abs/1810.10863) (Christopher Bowles, et al. 2018)

