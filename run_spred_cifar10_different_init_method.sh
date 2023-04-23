#!/bin/bash

weight_decay=1e-3
threshold=1e-3

# Define two arrays
first_array=("xavier_uniform_" "kaiming_uniform_" "kaiming_normal_")
second_array=("0" "1" "2")

# Combine the arrays using paste and iterate over the result
for init_red in independent sqrt
do
    paste <(printf "%s\n" "${first_array[@]}") <(printf "%s\n" "${second_array[@]}") \
    | while IFS=$'\t' read -r init_method cuda
    do
        (
            echo "$init_method is $cuda"
            python3 main.py --config configs/largescale/resnet18-spred-cifar10.yaml \
                            --weight-decay $weight_decay \
                            --name pretrain_weight_decay=$weight_decay-init=$init_method-red=$init_red \
                            --init_method $init_method \
                            --init_red $init_red \
                            --multigpu $cuda
            python3 main.py --config configs/largescale/resnet18-spred-cifar10.yaml \
                            --weight-decay $weight_decay \
                            --name pretrain_weight_decay=$weight_decay-init=$init_method-red=$init_red-finetune_threshold=$threshold \
                            --init_method $init_method \
                            --init_red $init_red \
                            --pretrained runs/resnet18-spred-cifar10/pretrain_weight_decay=$weight_decay-init=$init_method-red=$init_red/checkpoints/model_best.pth \
                            --threshold $threshold \
                            --multigpu $cuda
        ) &
    done
    wait
done

