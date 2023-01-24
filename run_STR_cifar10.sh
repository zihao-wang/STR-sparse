# for prune_rate in 0.99999 0.999999
# for prune_rate in 0.0 0.1 0.5 0.8 0.9 0.99 0.999 0.9999
# for weight_decay in 1e-3 5e-3
for weight_decay in 1e-3 2e-3 2.5e-3 3e-3 4e-3 5e-3
do
    python3 main.py --config configs/largescale/resnet18-str-cifar10.yaml \
                    --weight-decay $weight_decay \
                    --name weight_decay=$weight_decay \
                    --multigpu 2
done