# Beam-Splitter Transformer

This project aims to enhance the capability of transformer-based models like GPT to handle long sequences of input. The idea is inspired by the concept of a beam splitter in optics which splits a beam of light into two or more separate beams. In this context, we aim to "split" the attention mechanism of a transformer across different chunks of a long sequence, allowing the model to manage the complexity of large amounts of text. It might be something cool.

## Structure

- `data/` contains the raw data and processed data. This is typically not committed to git, but is necessary for understanding the project structure.
- `notebooks/` is for Jupyter notebooks used for data exploration and reporting. This is where we will keep our EDA (Exploratory Data Analysis) and model testing notebooks.
- `src/` holds the source code for the project:
    - `data/` contains scripts for downloading and processing data.
    - `models/` includes model definitions, training and inference scripts.
    - `utils/` holds utility scripts and functions used across the project.
- `tests/` is where we keep our unit tests to ensure code quality and correctness.
- `setup.py` is used for setting up the project's environment.

## Getting Started

To set up the environment:

```
pip install -r requirements.txt
```

Then, to run tests:

```
python -m unittest discover tests
```

## Progress

This is an ongoing project and we will update the README as we make progress.

## Contributions

We welcome contributions! Please see `CONTRIBUTING.md` for details on how to contribute.


