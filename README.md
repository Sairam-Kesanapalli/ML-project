# Statlog Shuttle Classification with Gaussian Mixture Models

A machine learning project that trains per-class Gaussian Mixture Models (GMMs) on the NASA Statlog Shuttle dataset for multi-class classification. The system learns probability distributions for each class and predicts new samples based on maximum likelihood.

## Project Overview

This project implements a generative classification approach using Gaussian Mixture Models. Instead of learning decision boundaries directly, it models the distribution of features for each class separately and classifies new samples by determining which class distribution they most likely belong to.

## Features

- **Per-Class GMM Training**: Trains separate Gaussian Mixture Models for each shuttle class
- **Automated Data Pipeline**: Fetches the Statlog Shuttle dataset from UCI ML Repository
- **Standardization**: Applies feature scaling for optimal GMM performance
- **External Prediction**: Make predictions on new data matching the training features
- **Model Persistence**: Saves trained models, scalers, and metadata for reuse
- **Test Data Generation**: Creates synthetic test data based on training distribution

## Dataset

The project uses the **Statlog Shuttle** dataset from the UCI Machine Learning Repository (ID: 148). This dataset contains:
- 9 numerical attributes describing shuttle radiator positions
- 7 classes representing different shuttle states
- Highly imbalanced classes (majority class ~80%)


## Installation

### Prerequisites

- Python 3.7+
- pip
- Git

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/Sairam-Kesanapalli/ML-project.git
cd ML-project
```

2. **Create and activate virtual environment** (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- pandas
- scikit-learn
- matplotlib
- joblib
- ucimlrepo

## Usage

### Automated Execution

Run the entire pipeline with a single command:

```bash
bash RUN.sh
```

This script will:
1. Clone/update the repository
2. Set up a virtual environment
3. Install all dependencies
4. Train the GMM models
5. Generate test data
6. Run predictions on test data

### Manual Execution

Navigate to the GMM directory:
```bash
cd GMM
```

#### 1. Train Models

```bash
python MAIN.py train
```

This will:
- Fetch the Statlog Shuttle dataset
- Split data into train/test sets (80/20 split)
- Train StandardScaler on training data
- Train separate GMMs for each class
- Save models to `output/` directory
- Evaluate performance on test set

Optional: Specify random state
```bash
python MAIN.py train --random-state 123
```

#### 2. Generate External Test Data

```bash
python Test_Data_gen.py
```

Creates `external/example_matched.csv` with 10 synthetic samples matching the training feature distribution.

#### 3. Predict on External Data

```bash
python MAIN.py predict --external external/example_matched.csv
```

Optional: Specify output path
```bash
python MAIN.py predict --external external/example_matched.csv --out output/my_predictions.csv
```

## Output Files

After training, the following artifacts are saved in `output/`:

- `scaler_gmm_shuttle.joblib` - Fitted StandardScaler
- `training_columns_shuttle.joblib` - Training feature column names
- `gmm_class_model_shuttle.joblib` - Dictionary of trained GMMs per class
- `external_gmm_predictions_matched.csv` - Predictions on external data

## Model Architecture

### Gaussian Mixture Model Configuration

- **Covariance Type**: Full (captures complete feature correlations)
- **Number of Components**: 
  - 1 component for classes with <50 samples
  - 2 components for larger classes
- **Maximum Iterations**: 300
- **Convergence**: Uses scikit-learn defaults

### Classification Process

1. Scale input features using fitted StandardScaler
2. Compute log-likelihood for each class's GMM
3. Assign sample to class with highest log-likelihood

## Performance

The model performance depends on the inherent separability of classes in the Statlog Shuttle dataset. Typical results:
- **Overall Accuracy**: ~95-99% on test set
- **Strengths**: Excellent on majority class (Class 1)
- **Challenges**: Lower recall on rare classes due to class imbalance

Check the console output after training for detailed classification metrics.

## Customization

### Using Your Own Data

To predict on your own CSV file:

1. Ensure your CSV has the same features as training data (9 numerical columns)
2. Features should be in the same order as the Statlog Shuttle dataset
3. Run: `python MAIN.py predict --external path/to/your/data.csv`

### Modifying GMM Parameters

Edit `MAIN.py` and adjust the `GaussianMixture` initialization:

```python
gmm = GaussianMixture(
    n_components=2,           # Number of Gaussian components
    covariance_type="full",   # Options: 'full', 'tied', 'diag', 'spherical'
    random_state=random_state,
    max_iter=300              # Maximum EM iterations
)
```

## Troubleshooting

**Error: "ucimlrepo not installed"**
- Install via: `pip install ucimlrepo`

**Error: "Required artifacts not found"**
- Run `python MAIN.py train` before attempting predictions

**Error: "training_columns_shuttle.joblib not found"**
- Run training first to generate required artifacts

**Poor performance on rare classes**
- This is expected due to extreme class imbalance in the dataset
- Consider SMOTE, class weights, or ensemble methods for improvement

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- UCI Machine Learning Repository for the Statlog Shuttle dataset
- scikit-learn for GMM implementation
- NASA for the original shuttle data

## Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Note**: This is an educational project demonstrating generative classification with GMMs. For production use, consider additional validation, hyperparameter tuning, and handling of edge cases.
