import torch
import os

from torchvision import datasets, transforms

from model_eval.resnet_eval import ResNet18

from utils.logging import AverageMeter
from utils.eval_utils import accuracy
import tqdm


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=int)
parser.add_argument("--thr", type=float, default=1e-3)
parser.add_argument("--data", type=int, default=10)
parser.add_argument("--method", type=str, default="VanillaConv")

class CIFAR10:
    def __init__(self):

        data_root = os.path.join("rawdata", "cifar10")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": 20, "pin_memory": True} if use_cuda else {}


        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = datasets.CIFAR10(
            data_root,
            True,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=True
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=256, shuffle=True, **kwargs
        )

        self.val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                data_root,
                False,
                transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
                download=True
            ),
            batch_size=256,
            shuffle=False,
            **kwargs
        )


class CIFAR100:
    def __init__(self):

        data_root = os.path.join("rawdata", "cifar10")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": 20, "pin_memory": True} if use_cuda else {}


        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = datasets.CIFAR100(
            data_root,
            True,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=True
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=256, shuffle=True, **kwargs
        )

        self.val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                data_root,
                False,
                transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            ),
            batch_size=256,
            shuffle=False,
            **kwargs
        )


cifar10_dataset = CIFAR10()
cifar100_dataset = CIFAR100()

def evaluate_checkpoint(ckpt_path = "runs/resnet18-l1-cifar10/l1=1e-3/prune_rate=0.0/0/checkpoints/model_best.pth",
                        dataset = "cifar10",
                        conv_type = "vanilla",
                        thr = 1e-3,
                        device="cuda:0",
                        num_classes=100):

    model = ResNet18(conv_type=conv_type,num_classes=num_classes)
    state_dict = torch.load(ckpt_path)
    model_state_dict = {}

    for k, v in state_dict['state_dict'].items():
        model_state_dict[k[7:]] = v

    model.load_state_dict(model_state_dict)

    for n, m in model.named_modules():
        if hasattr(m, 'thr'):
            m.thr = thr
            print("setting threshold", n)

    data = cifar10_dataset if dataset == 'cifar10' else cifar100_dataset
    val_loader = data.val_loader

    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    # switch to evaluate mode
    model.eval()

    model.to(device)

    with torch.no_grad():
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            images = images.to(device)
            target = target.to(device).long()

            # compute output
            output = model(images)


            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

    acc = top1.avg

    nonzero_sum, total_sum = 0, 0
    for n, m in model.named_modules():
        if hasattr(m, 'getSparsity'):
            nonzero, total, _= m.getSparsity(thr)
            nonzero_sum += nonzero
            total_sum += total
    compression_ratio = total_sum / nonzero_sum

    # return {"acc": acc, "cr": compression_ratio.item()}
    return compression_ratio.item(), acc




# evaluate_checkpoint(
# ckpt_path = "runs/resnet18-spared-cifar100/weight_decay=3e-4/prune_rate=0.0/checkpoints/model_best.pth",
# dataset = "cifar100",
# conv_type = "VanillaConv1",
# thr = 1e-3,
# device="cuda:0",
# num_classes=100
# )

args = parser.parse_args()

evaluate_checkpoint(
    ckpt_path=args.ckpt,
    dataset=f"cifar{args.data}",
    conv_type=args.method,
    thr=args.thr,
    device="cuda:0",
    num_classes=args.data
)