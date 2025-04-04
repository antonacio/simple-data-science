# Simple Data Science

**This project compiles simple and practical examples for common data science use cases with tabular data.**

You can access the complete examples using the following links:
1. [Binary Classification](https://antonacio.github.io/simple-data-science/classification-binary.html)
2. [Multiclass Classification](https://antonacio.github.io/simple-data-science/classification-multiclass.html)
3. [Regression](https://antonacio.github.io/simple-data-science/regression.html)
4. [Clustering](https://antonacio.github.io/simple-data-science/clustering.html)
5. [Histogram Analysis](https://antonacio.github.io/simple-data-science/histogram_analysis.html)

## Setup

### Environment

In this repository, we use UV—a handy Python package and project manager. To install UV, follow [these instructions](https://docs.astral.sh/uv/getting-started/installation/).

To set up the environment and install the required dependencies, run the following commands in your terminal:

```bash
cd simple-data-science     # change to the project's directory
uv venv --python 3.12      # create a python 3.12 virtual environment
source .venv/bin/activate  # activate virtual environment
uv sync                    # synchronize dependencies
pre-commit install         # install pre-commit hooks
```

If you want to deactivate and delete the virtual environment, run:

```bash
deactivate                 # deactivate virtual environment
rm -rf .venv               # delete virtual environment
```

### Data

The examples in this project use the publicly available [Fetal Health Dataset](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) and [Medical Insurance Payout Dataset](https://www.kaggle.com/datasets/harshsingh2209/medical-insurance-payout).

Because the datasets are small, they are available as `.zip` files in the repository's `data/` folder. You can unzip them with your preferred software or simply run `make unzip-datasets` in your terminal.

## Data Science Use Cases

### Code

The code for the data science use cases described above is located in the `src/` directory. To run a use case, open the corresponding notebook file, select your preconfigured virtual environment, and execute the cells. Have fun!

To update the HTML files with the latest notebook outputs, run the command `make convert-notebooks-to-html` in your terminal.

### Contributions

We welcome contributions of all kinds. Whether you have questions, spot a bug, or want to enhance the code, documentation, or tests, please feel free to start a discussion or open a pull request. Your feedback, ideas, and fixes are vital in making this project better for everyone!

### License

MIT
