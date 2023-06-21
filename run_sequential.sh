#!/bin/bash

# chunk 1
# adam optimizer
#python train_facial_expression.py --cartoon_prec 0.0 && \
#python train_facial_expression.py --cartoon_prec 0.2 && \
#python train_facial_expression.py --cartoon_prec 0.4 && \
#python train_facial_expression.py --cartoon_prec 0.6 && \

# adam optimizer
#python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.0 && \
#python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.2 && \
#python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.4 && \
#python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.6 && \

# sgd (reduce) optimizer
#python train_facial_expression.py --optimizer sgd --cartoon_prec 0.0 && \
#python train_facial_expression.py --optimizer sgd --cartoon_prec 0.2 && \
#python train_facial_expression.py --optimizer sgd --cartoon_prec 0.4 && \
#python train_facial_expression.py --optimizer sgd --cartoon_prec 0.6

# chunk 2
# try adam, lr=0.0001, smaller/bigger batch
#python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.0 --batch_size 32 && \
#python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.0 --batch_size 128 && \
# try test with cartoon
#python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 1.0 --test_mode cartoon && \
#python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.8 --test_mode cartoon && \
#python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.6 --test_mode cartoon && \
#python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.4 --test_mode cartoon && \
#python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.2 --test_mode cartoon && \
#python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.0 --test_mode cartoon

# chunk 3
# sgd (reduce) optimizer, smaller/bigger batch
#python train_facial_expression.py --optimizer sgd --cartoon_prec 0.0 --batch_size 32 && \
#python train_facial_expression.py --optimizer sgd --cartoon_prec 0.0 --batch_size 128 && \
#python train_facial_expression.py --optimizer sgd --cartoon_prec 0.2 --batch_size 32 && \
#python train_facial_expression.py --optimizer sgd --cartoon_prec 0.2 --batch_size 128 && \
# train on united dataset
#python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.0 --batch_size 32 --train_on_united yes && \
#python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.0 --batch_size 64 --train_on_united yes && \
#python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.0 --batch_size 128 --train_on_united yes && \
#python train_facial_expression.py --optimizer sgd --cartoon_prec 0.0 --batch_size 32 --train_on_united yes && \
#python train_facial_expression.py --optimizer sgd --cartoon_prec 0.0 --batch_size 64 --train_on_united yes && \
#python train_facial_expression.py --optimizer sgd --cartoon_prec 0.0 --batch_size 128 --train_on_united yes

# chunk 4
# take best model so far and try less than 10% cartoon
python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.05 --batch_size 32 && \
python train_facial_expression.py --lr_adam 0.0001 --cartoon_prec 0.02 --batch_size 32 && \
#