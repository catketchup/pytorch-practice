# PyTorch Practice

A hands-on, progressive set of exercises to learn PyTorch from the ground up.

## Setup

```bash
pip install -r requirements.txt
```

## How It Works

Each exercise file contains `TODO` markers where you fill in the code. When you're done, run the file — if all assertions pass, you've solved every exercise!

### Python scripts

```bash
# Run an exercise
python exercises/01_tensors.py

# Stuck? Check the solutions
python solutions/01_tensors.py
```

### Jupyter notebooks

Interactive notebooks are also available in the `notebooks/` directory. Open them with:

```bash
jupyter notebook notebooks/
```

## Learning Path

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 01 | **Tensors** | Creation, dtypes, indexing, reshaping, broadcasting, math ops, NumPy bridge |
| 02 | **Autograd** | `requires_grad`, `.backward()`, gradient accumulation, `no_grad`, `detach`, custom Functions |
| 03 | **Datasets & DataLoaders** | `TensorDataset`, custom `Dataset`, batching, transforms, collate functions, train/val splits |
| 04 | **nn.Module** | `Linear`, `Sequential`, custom modules, forward pass, loss functions, parameter inspection, freezing layers |
| 05 | **CNNs** | Conv1d/2d, pooling, output size formula, batch norm, dropout, building full CNN architectures |
| 06 | **Training Loop** | Optimizers, SGD/Adam, mini-batch training, LR scheduling, gradient clipping, classification |
| 07 | **Transfer Learning & Real Patterns** | Saving/loading models, fine-tuning, early stopping, validation loops, device management, reproducibility |

## Exercises are Progressive

Each file builds on concepts from the previous ones. Work through them in order:

```
01 → 02 → 03 → 04 → 05 → 06 → 07
```

## Tips

- Read the docstrings and comments in each exercise carefully — they contain the instructions
- Start with the first `TODO` and work downward
- Use `print()` to inspect intermediate results
- If an assertion fails, read the error message — it tells you exactly what was expected
- Only peek at solutions when you're truly stuck