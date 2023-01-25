# for prune_rate in 0.99999 0.999999
# for prune_rate in 0.0 0.1 0.5 0.8 0.9 0.99 0.999 0.9999
# for weight_decay in 1e-3 5e-3
for weight_decay in 1e-5 3e-5 1e-4 3e-4 1e-3
do
    python3 main.py --config configs/largescale/resnet18-spared-cifar10.yaml \
                    --weight-decay $weight_decay \
                    --name weight_decay=$weight_decay \
                    --multigpu 0
done