# Assignment 2 - PyTorch MLP Grid Search

# Overview 
Trains a neural network on the CIFAR-100 image dataset multiple times, each time with a different combination of settings, and records how accurate each one was. Results are used to determine the best settings for the neural network that recognizes images from the CIFAR-100 dataset.

## Dataset
- CIFAR-100 (using coarse labels)
  - 50,000 training images 
  - 10,000 test images 
  - Each image is 32×32 pixels with 3 colors: red, green, and blue 
  - Images belong to 20 broad categories (animals, vehicles, furniture, etc.)

## File Structure 
Assignment_2/

├── grid_search.py          
├── train                   
├── test                   
└── grid_search_output/   
    └── results.csv  

## Steps to run 
1. Import the `train` and `test` files into the same folder directory as `grid_search.ipynb` or `grid_search.py`
2. Run `python grid_search.py` or cell if running the Jupyter Notebook
3. Note: On Apple MacBook M1, this will take hours to run
4. Results will be saved to `grid_search_output/results.csv`

## Outputs 
Each row in the CSV is one training run. Two main columns to review
- `accuracy` - Explains how well the model did on the training data
- `val_accuracy` - Explains how well the model did on the test data

## Model Configuration Settings
This table explains key configuration settings used in a neural network model.

| Setting               | What It Means                                      | Options                             |
|----------------------|----------------------------------------------------|-------------------------------------|
| `units`              | How many "brain cells" (neurons) per layer         | `120` or `240` neurons              |
| `hidden_activations` | How neurons process information                    | `ReLU` (fast) or `Sigmoid` (smooth) |
| `activation`         | How final answer is formatted                      | `Softmax` or `Sigmoid`              |
| `loss`               | How we measure mistakes                            | `MSE` or `Categorical Crossentropy` |
| `optimizer`          | How we learn from mistakes                         | `Adam` or `Adagrad`                 |
| `batch_size`         | How many images to process at once                 | `1000` or `2000`                    |

## Model Overview
- Input Features: 3072 (32×32 pixels × 3 color channels)
- Hidden Layers: 5
- Output: 20 (CIFAR-100 coarse labels)
- Training Epochs: 200 per run
- Total Combinations: 64 (2×2×2×2×2×2)

### Accuracy Expectations
- MLPs are not best suited for images. As noted in the instructions, anything about 20% is considered passing for this assignment

----
# Setup Instructions

## 1. Install Required Packages
```bash
pip install torch numpy pandas
```

## 2. Download the Datasets
Place the following two files in the same folder as the main code:
- `tain` - Training data file
- `test` - Test data file

## 3. Run  the Code
```bash
python grid_search.py
```

## High Level - Code Logic 
1. Loading: The program loads 50,000 training images and 10,000 test images 
2. Testing: It tries all 64 combinations of settings 
3. Training: For each combination, it trains the network for 200 rounds (epochs)
4. Saving: Results are saved to `grid_search_output/results.csv` 
5. Display: Shows all results sorted from best to worst

---
# Code Breakdown

## 1. One-hot Encoding
Converts class labels (0-19) into binary vectors for MSE loss:

```python
y_train_oh = np.eye(num_classes, dtype='float32')[y_train]
```

# 2. Build the Model
Create a neural network: 
- 5 hidden layers
- Configure units per layer
- Configure activation functions 
- Output layer with 20 classes

# 3. Run the Model
For each parameter combination:
- Builds the model 
- Trains for 200 epochs 
- Calculates accuracy on both training and test data 
- Returns results

# 4. Grid Search
Loops through all 64 combinations of parameters:

```python
for combo in combinations:
    params = dict(zip(keys, combo))
    metrics = train_and_evaluate(...)
```

# 5. Write to CSV
Save the results after each run to prevent data loss:

```python
with open('grid_search_output/results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
```

# 6. Display Results
Save and show the results from best to worst, ranked by validation accuracy:

```python
for epoch in range(200):
```

---

# Troubleshooting
| Problem               | Solution                                                      |
|----------------------|--------------------------------------------------------------|
| File not found error | Ensure train and test files are in the same directory        |
| Out of memory        | Reduce `batch_size` or `units` in the parameter dictionary   |
| Takes too long       | Reduce epochs from 200 to 50 for initial testing             |
| Low accuracy         | Normal for MLPs on images — 20–30% is acceptable             |


