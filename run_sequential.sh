#!/bin/bash

# adam optimizer
python train_facial_expression.py --cartoon_prec 0.0 && \
python train_facial_expression.py --cartoon_prec 0.2 && \
python train_facial_expression.py --cartoon_prec 0.4 && \
python train_facial_expression.py --cartoon_prec 0.6 && \

# adam optimizer
python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.0 && \
python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.2 && \
python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.4 && \
python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.6 && \

# sgd (reduce) optimizer
python train_facial_expression.py --optimizer sgd --cartoon_prec 0.0 && \
python train_facial_expression.py --optimizer sgd --cartoon_prec 0.2 && \
python train_facial_expression.py --optimizer sgd --cartoon_prec 0.4 && \
python train_facial_expression.py --optimizer sgd --cartoon_prec 0.6
