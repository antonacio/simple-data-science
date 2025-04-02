[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://github.com)

# Simple Data Science

This project compiles simple and practical examples for common Data Science use cases with tabular data.

You can access complete examples using the following links:
1. Binary Classification
2. Multiclass Classification
3. Regression
4. Clustering
5. Histogram Analysis

## Setup

```bash
make setup
```

or run the following commands sequentially:

```bash
pip install --upgrade uv
uv venv --python 3.12
source .venv/bin/activate
uv sync
pre-commit install
```

## Data

The examples in this project use the publicly available [Fetal Health Dataset](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) and [Medical Insurance Payout Dataset](https://www.kaggle.com/datasets/harshsingh2209/medical-insurance-payout).

Because the datasets are small, they are available as `.zip` files in the repository's `data/` folder. You can unzip them with your preferred software or simply run `make unzip-datasets` in your terminal.

## Contributions

We welcome contributions of all kinds! Whether you have questions, spot a bug, or want to enhance the code, documentation, or tests, please feel free to start a discussion or open a pull request. Your feedback, ideas, and fixes are vital in making this project better for everyone!

## License

MIT
