# ValidityNovelty-Prediction (not finished, yet)
University project in the subject "Deep Learning for Natural Language Processing". The task is to classify for a set of triples containing a topic, a premise, and a conclusion whether the conclusion is valid and/or novel given the premise. The task alongside the original GitHub Repo and paper are mentioned here: https://phhei.github.io/ArgsValidNovel/

## Getting Started
First, clone the repository:
```bash
git clone https://github.com/hannesill/Validity-Novelty-Classification.git
```

This project was done using Python 3.10. All other requirements are listed in the requirements.txt file. To install them, run
```bash
pip install -r requirements.txt
```

## Usage

### Train RNN (Baseline)
To run the baseline RNN (LSTM) model, run
```bash
python3 rnn.py
```

### Train Transformer (Baseline)
To run the baseline transformer model, run
```bash
python3 transformer.py
```

### Train Transformer (Enhanced)
To run the enhanced approach with the transformer model, run
```bash
python3 transformer.py --preprocess
```

### Optional Data Augmentation
Optionally, you can set the --augment flag in addition to the --preprocess flag. However, this does not improve the model, and takes a lot of time. I just left it in the code for potential future augmentation modifications. This is not part of my enhanced approach.

### Evaluation
After training a model, the model gets tested on the test set and writes the results to a prediction .csv file. To evaluate the model, run
```bash
python3 Evaluator.py A PATH_TO_TEST_SET PATH_TO_PREDICTION_FILE
```
where A is the subtask binary validity-novelty-classification that this project is about. The task is linked at the top of this README.
