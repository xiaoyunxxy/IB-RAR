# IB-RAR
This is the repo for our work entitiled `IB-RAR: Information Bottleneck as Regularizer for Adversarial Robustness`.

## Environment
```bash
torch==1.12.1
torchattacks==3.3.0
torchvision==0.13.0
tqdm==4.64.1
numpy==1.23.4
```

## Running
`python train_AT.py --network vgg16 --dataset cifar10 --mi_loss 1 --fc 1`
