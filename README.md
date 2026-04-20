# Self-Pruning Neural Network

PyTorch implementation of the Tredence AI Intern case study on CIFAR-10.

## What is implemented

- Custom `PrunableLinear(in_features, out_features)` layer
- Learnable `gate_scores` for each weight
- Gated weight computation:
`gates = sigmoid(gate_scores)`  
`pruned_weights = weight * gates`
- Training loss:
`total_loss = cross_entropy + lambda * L1(gates)`
- Multi-lambda experiment runner
- Test accuracy + sparsity reporting
- Gate distribution plot for the best model

## Files

- `self_pruning_cifar10.py`: full training/evaluation script
- `requirements.txt`: dependencies

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python self_pruning_cifar10.py --epochs 20 --batch-size 128 --lambdas 1e-6,1e-5,1e-4 --output-dir outputs
```

## Generated outputs

- `outputs/results_summary.csv`
- `outputs/results_table.md`
- `outputs/best_gate_distribution.png`
- `outputs/lambda_*/best_model.pt`

## Report table template

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|---:|---:|---:|
1e-04 | 58.70 | 88.97 
| 5e-04 | 55.67 | 98.51
| 1e-03 | 52.33 | 99.56 
