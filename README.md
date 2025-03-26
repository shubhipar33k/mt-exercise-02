# MT Exercise 2: RNN Language Model - Alice in Wonderland Edition 

## Task 1

This project trains a recurrent neural network (RNN) language model using PyTorch on **"Alice’s Adventures in Wonderland"** from Project Gutenberg.

## Dataset

- **Source**: Project Gutenberg, [https://www.gutenberg.org/ebooks/11](https://www.gutenberg.org/ebooks/11)
- **Preprocessing**:
  - Removed Gutenberg headers/footers
  - Normalized whitespace and cleaned characters
  - Tokenized using `sacremoses`
  - Split into `train.txt`, `valid.txt`, `test.txt`
- **Final dataset size**: 2955 training lines (~2600 vocab size)

## Code Adjustments

Due to changes in PyTorch 2.6, the model loading step fails unless full deserialization is explicitly allowed.

To fix this, we made the following small code edits:

- In `main.py`:
  Changed  
  `model = torch.load(f)`  
  to  
  `model = torch.load(f, weights_only=False)`

- In `generate.py`:
  Changed  
  `model = torch.load(f, map_location=device)`  
  to  
  `model = torch.load(f, map_location=device, weights_only=False)`

These changes ensure that the full model class (`RNNModel`) can be deserialized after training and during text generation.


## Training Setup

- **Model**: Word-level RNN (from PyTorch examples)
- **Embedding size**: 200
- **Hidden size**: 200
- **Dropout**: 0.5
- **Epochs**: 40

### Perplexities

- **Validation PPL**: 2.55
- **Test PPL**: 2.53

## Sample Output

"from keeping in her fall . <eos> The master is , , I mean--I waxworks ! " <eos> DUCHESS I should said you cheat dance and the exclamations . <eos> If you know , you never just if pleases of the dance . <eos> [ _ Music grins away . _ ] Divide your II help ! <eos> The jurymen was an old deep , not to lose you such a joke . <eos> It 's it . <eos> Treacle ! <eos> Oh ! <eos> Where your nice Nonsense ! <eos> it 's for the tea . <eos>"


## How to Run

```bash
# Train
./scripts/train.sh

# Generate
./scripts/generate.sh

# Sample output saved to:
samples/sample
```

##  Task 2: Dropout Experiments

The models were trained on the *Alice in Wonderland* dataset prepared in Task 1.

## Dropout Values Tested:
- 0.0
- 0.2
- 0.3
- 0.5
- 0.6

#### Training Setup:
- Embedding size: 250
- Hidden units: 250
- Epochs: 40
- Optimizer: SGD with manual learning rate step
- Tokenized with Sacremoses
- Vocabulary size: 2600

#### Results Summary:

| Dropout | Test Perplexity |
|---------|------------------|
| 0.0     | **1.17** Best    |
| 0.2     | 1.69             |
| 0.3     | 2.05             |
| 0.5     | 5.39             |
| 0.6     | **9.98** Worst   |

#### Visualizations:
- Perplexity line plots are saved in `plots/`
- Epoch-wise perplexities in `tables/train_perplexity.csv` and `tables/valid_perplexity.csv`

#### Observations:
- The model with **dropout = 0.0** performed best on the test set.
- Higher dropout values led to over-regularization and poor performance.
- The validation perplexity curves confirm that moderate dropout helps early on, but excessive dropout hurts convergence.

#### Text Sample: Best vs Worst

**Dropout 0.0 (Best model):**
> Victor Alice in Wonderland W. costumes McCLURG Alice … ALICE There’s a mistake somewhere — WHITE QUEEN

**Dropout 0.6 (Worst model):**
> Treacle! Oh! Where your bone! DUCHESS … It’s wake into the eight. KNAVE

The best model generated more stylistically coherent and content-relevant output, with proper nouns and phrases that reflect the training corpus.
