# ITILA_LTH
#  Lottery Ticket Hypothesis — PyTorch Implementation



This repository provides a **PyTorch implementation** of the paper  
 *“The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks”*  
by **Jonathan Frankle** and **Michael Carbin**, presented at **ICLR 2019**.  
([Read the paper](https://arxiv.org/abs/1803.03635))

The code implements **iterative magnitude pruning (IMP)** to identify “winning tickets” — sparse subnetworks that, when trained from their original initialization, match the accuracy of the dense model.

---

##  Overview

The **Lottery Ticket Hypothesis (LTH)** proposes that within large, randomly initialized networks, there exist smaller sparse subnetworks (“winning tickets”) that can train in isolation to achieve comparable performance to the full model.

This repository allows you to:
- Train → Prune → Reset → Retrain neural networks iteratively.
- Compare two modes:
  - `lt`: Reset surviving weights to **original initialization**.
  - `reinit`: Reset surviving weights to **new random values**.
- Visualize pruning progress, sparsity, and accuracy across iterations.

---

##  Requirements

Install dependencies using:

```bash
pip3 install -r requirements.txt
```

**Main dependencies**
- Python ≥3.7
- PyTorch 1.2.0, Torchvision 0.4.0
- NumPy, Matplotlib, Seaborn
- TensorBoardX, tqdm

---

##  How to Run

Example command:

```bash
python3 main.py --prune_type=lt --arch_type=fc1 --dataset=mnist --prune_percent=10 --prune_iterations=35
```

### Available Arguments

| Argument | Description | Default |
|-----------|--------------|----------|
| `--prune_type` | `lt` (Lottery Ticket) or `reinit` (Random Reinit) | `lt` |
| `--dataset` | `mnist`, `fashionmnist`, `cifar10`, `cifar100` | `mnist` |
| `--arch_type` | `fc1`, `lenet5`, `alexnet`, `vgg16`, `resnet18`, `densenet121` | `fc1` |
| `--prune_percent` | % of weights pruned per cycle | `10` |
| `--prune_iterations` | Number of pruning rounds | `35` |
| `--lr` | Learning rate | `1.2e-3` |
| `--batch_size` | Training batch size | `60` |
| `--end_iter` | Epochs per pruning round | `100` |
| `--gpu` | Which GPU to use | `0` |

---

##  Adding New Models or Datasets

### ▶ Adding a New Architecture
1. Create a file in `/archs/<dataset_name>/new_model.py`.
2. Define your PyTorch model in it.
3. Import the model in `main.py` under the appropriate dataset section.
4. Add entry:
   ```python
   elif args.arch_type == "new_model":
       model = new_model.MyModel().to(device)
   ```

### ▶ Adding a New Dataset
1. Create `/archs/new_dataset/` and add compatible model files.
2. Add loading logic in `main.py` similar to existing datasets.
3. Ensure image size, channels, and classes match the architecture.

---

##  Combining Plots

After generating model outputs for both `lt` and `reinit`:

```bash
python3 combine_plots.py
```

This script combines and compares the test accuracy vs. sparsity curves for both methods.  

