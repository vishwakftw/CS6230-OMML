#!/bin/bash

cuda=$1

for arch in ` ls ../ARCHS/Autoencoders `;
    do    
    for opt in 'adadelta' 'adagrad' 'adam' 'cm' 'nag' 'rmsprop';
        do
        mkdir $opt
        for opt_params in ` ls ../OPT_JSONS/$opt `;
            do
            python3 MLPAutoencoder.py --opt $opt --opt_params ../OPT_JSONS/$opt/$opt_params \
                                      --architecture ../ARCHS/Autoencoders/$arch --init xavier --dataset mnist \
                                      --dataroot ../Datasets/MNIST --cuda $cuda --maxiter 10000;
            done;
        done;
    done
