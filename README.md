# The TABLE Group's Transformer!

The repo for the mini replication of transformer and its translation task for CS6741 Class Replication Project

## Installation
We recommend using the virtual environment. Please install the below packages before running the codes

```
pip install -r requirements.txt
python -m spacy download en
python -m spacy download de
```

## File Description
### model component file
transformer_model.py
List of Functions:
- mask: three different types of masks
1. create_padding_mask
2. create_look_ahead_mask
3. create_mask

- attention: attention calculation
1. scaled_dot_product_attention: generic attention calculation
2. MultiHeadAttention

- encoder:
1. PositionwiseFeedForward
2. EncoderLayer
3. Encoder

- decoder:
1. DecoderLayer
2. Decoder

- assemble models:
transformer

### training file
training.py

List of Functions:

- Loss:
- 
We have experimented with different loss functions, including *crossentropy*, *NNLoss*, and *KLNDiv*. We reported our results in our main findings

- Dynamic_LR_Scheduler:
- 
We will illustrate the hyperparameters that we have used in. This function returns step, learning rate, etc 

- LabelSmoothing:
We take the label smoothing as an approach to penalize the model when it is ``over-confident"

## Training
- build_vocab

- data_process

- generate_batch

- train

- evaluate

- epoch_time

### Hyperparameters

1. *Batch Size*

In our experimental settings, we found the bottleneck is mainly the batch size. However, due to the bottleneck of our GPU resources, we cannot set the batch size as illustrated in the Google original paper.
We experimented with the batch size in (5, 10, 15)

2. *Learning rate*

We followed the original setting of learning rate computing and set the factor=1 as the same in the original paper.  

```
self.factor * \
            (self.d_model ** (-0.5) *
            min(self.step_num ** (-0.5), self.step_num * self.warmup_step ** (-1.5)))
```

3. *Label Smoothing*

We experimented with the setting of using label smoothing with a smooth ratio to be 0.1 v.s. 0, we reported our findings of the label smoothing factor of the model's training loss and perplexity. 

 *Others*
 
N_stack = 6

d_model = 512

num_heads = 8

d_ff = 2048

dropout = 0.1

beta1 = 0.9

beta2 = 0.98

epsilon = 10**(-9)

warmup_step = 4000


### Findings and Results
Please refer to findings.pdf for more details. 

## Summary
