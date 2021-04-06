# The TABLE Group's Transformer!

The repo for the mini replication of transformer and its translation task for CS6741 Class Replication Project

## Installation
We recommend to use the virtual environment. Please install the below packages before running the codes

```
pip install -r requirements.txt
python -m spacy download en
python -m spacy download de
```

## File Description
### model component file
transformer_model.py
Function lists:
- mask: three different types of masks
1. create_padding_mask
2. create_look_ahead_mask
3. create_mask

- attention:
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


### 

## Training

## Results

## Summary
