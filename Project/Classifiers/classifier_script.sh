#!/bin/bash

cuda=$1
dset=$2
droot=$3
s=0
e=0

if [ "$dset" == "mnist" ] ; then
    s=1
    e=4
fi

if [ "$dset" == "cifar10" ] ; then
    s=5
    e=8
fi

if [ "$dset" == "svhn" ] ; then
    s=5
    e=8
fi

for index in ` seq $s $e `;
    do    
    for opt in 'adadelta' 'adagrad' 'adam' 'cm' 'nag' 'rmsprop';
        do
        mkdir $opt
        for opt_params in ` ls ../OPT_JSONS/$opt `;
            do
            python3 Classifier.py --opt $opt --opt_params ../OPT_JSONS/$opt/$opt_params \
                                  --architecture ../ARCHS/Classifiers/arch_$index.json --init xavier --dataset mnist \
                                  --dataroot $droot --cuda $cuda --maxiter 10000;
            done;
        done;
    done
