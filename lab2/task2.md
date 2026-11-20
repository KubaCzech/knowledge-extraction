# L2 Labs: Tabular Datasets

1. My email address: rafal.bachorz@put.poznan.pl

1. Commit all results to the repository via a pull request (PR), in `sessions\L2_tabular_datasets\lab\results\your_subdirectory` (see the README.md file in the root directory for the PR rules).

1. Set up a Python environment with, for example, the following libraries:

    * pandas: for data analysis
    * xgboost: for predictive modeling
    * catboost: for predictive modeling
    * matplotlib: for data visualization
    * scikit-learn: for creating data-preprocessing pipelines

1. Consider the following classic datasets:

    * **Iris Dataset**
        * **Description**: The Iris dataset is one of the most famous datasets used for classification. It contains 150 samples from three species of iris flowers, with four features (sepal length, sepal width, petal length, and petal width) used to classify the species.
        * [Link to dataset](https://archive.ics.uci.edu/ml/datasets/iris)

    * **MNIST Dataset**
        * **Description**: The MNIST dataset consists of 70,000 images of handwritten digits (0–9). It is commonly used for image classification and is widely regarded as a benchmark for machine learning models, particularly in computer vision.
        * [Link to dataset](http://yann.lecun.com/exdb/mnist/)

    * **Boston Housing Dataset**
        * **Description**: This dataset contains information about housing prices in the suburbs of Boston. The goal is to predict the price based on 13 features, such as crime rate and number of rooms. It is commonly used for regression problems.
        * [Link to dataset](https://github.com/selva86/datasets/blob/master/BostonHousing.csv)

    * **Wine Dataset**
        * **Description**: The Wine dataset contains data on the chemical composition of wines from three different cultivars. The task is to classify wines based on 13 chemical features, making this a popular dataset for classification.
        * [Link to dataset](https://archive.ics.uci.edu/ml/datasets/wine)

    * **Breast Cancer Wisconsin Dataset**
        * **Description**: This dataset contains features extracted from digitized images of breast masses. The task is to classify whether a tumor is malignant or benign based on these features. It is widely used for binary classification.
        * [Link to dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

    * **Titanic Dataset**
        * **Description**: The Titanic dataset is a classic dataset used to predict passenger survival based on features such as age, class, and sex. It is often used for binary classification.
        * [Link to dataset](https://www.kaggle.com/c/titanic/data)

    * **CIFAR-10 Dataset**
        * **Description**: The CIFAR-10 dataset consists of 60,000 32 × 32 color images in 10 classes, with 6,000 images per class. It is widely used to train computer vision models, especially for image classification.
        * [Link to dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

    * **Adult Income Dataset**
        * **Description**: The Adult dataset contains demographic information from the 1994 U.S. Census. The task is to predict whether an individual earns more than $50,000 per year based on attributes such as age, education, and occupation. It is commonly used for binary classification.
        * [Link to dataset](https://archive.ics.uci.edu/ml/datasets/adult)

    * Choose one dataset of your choice for either a classification or regression task, and use it for further exploratory and predictive analysis.

1. Carry out exploratory data analysis (EDA):

    * Identify the independent variables (features) and the dependent variable (target).
    * Provide a basic statistical analysis of the data:
        * Measures of central tendency, such as the mean, median, and mode
        * Measures of variability, such as the range, standard deviation, coefficient of variation, and interquartile distance
        * Measures of distribution shape, such as skewness and kurtosis
    * Are there outliers in the data? How can you detect them?
    * Plot histograms of selected features. What do they show? Are there any unexpected findings? Describe any patterns you observe in the data.
    * Plot boxplots and violin plots of selected features or targets. Describe any patterns you observe in the data.

1. Create a Scikit-learn `Pipeline` object for data processing and model application:

    * Design a Scikit-learn preprocessing `Pipeline` that includes `StandardScaler`, `VarianceThreshold` (and other components if needed), and the XGBoost and CatBoost models.
    * Split the dataset into training and test sets.
    * Fit the pipeline using the training data.

1. Estimate the quality of the models:

    * Apply the fitted `Pipeline` object to the test data.
    * Estimate model quality using relevant metrics.
    * Compare the predictive performance of the two models.
    * Collect all data and observations in a Jupyter Notebook.

1. Potential further improvements:

    * Does including `StandardScaler` affect the quality of the Gradient Boosting models?
    * How would you extend the pipeline to include dimensionality reduction using Principal Component Analysis (PCA)?
    * What else should be done before deploying the models to a production environment?

1. Tabular Data Challenge

    * In `sessions\L2_tabular_datasets\lab\challenge`, you will find anonymized data for `solubility` (one of the ADMET properties) as a function of molecular descriptors:
        * `TrainingData_Blinded.csv`: the training data, containing both the independent variables (X1–X339) and the dependent variable (Y)
        * `TestData_Blinded_NoY.csv`: the test data, containing only the independent variables (X1–X339), which have the same meanings as the corresponding variables in the training data file
    * The task is to create the best-performing model (using a chosen Gradient Boosting method) for predicting the dependent variable (Y), and then apply the best model to the test data.
    * The metric used to compare your predictions with the ground-truth targets will be RMSE.
    * Commit your predictions via a PR to the appropriate subdirectory as a CSV file: `sessions\L2_tabular_datasets\lab\results\your_subdirectory\challenge.csv`. The file should contain two columns: `Name` (an index, as in the `TestData_Blinded_NoY.csv` file) and `Y_pred` (the prediction).

1. Store results only in `sessions\L2_tabular_datasets\lab\results\your_subdirectory`. Include the solutions for tasks 1–8 in a single Jupyter Notebook named `tasks_1_8.ipynb`. Save the challenge work in the same directory in a notebook named `challenge.ipynb`.

1. Submission deadline: 6.11.2026.
