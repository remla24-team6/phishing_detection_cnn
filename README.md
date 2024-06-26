![Test Status](https://img.shields.io/badge/tests-passing-brightgreen)
![Adequacy Score](https://img.shields.io/badge/adequacy_score-1.00-blue)
![Train Accuracy](https://img.shields.io/badge/train_accuracy-0.5084999799728394-blue)
![Train Loss](https://img.shields.io/badge/train_loss-0.6965246796607971-blue)
![Validation Accuracy](https://img.shields.io/badge/val_accuracy-0.45010000467300415-blue)
![Validation Loss](https://img.shields.io/badge/val_loss-0.6957074999809265-blue)
![Test Accuracy](https://img.shields.io/badge/test_accuracy-0.6022-blue)
![Average Precision](https://img.shields.io/badge/avg_precision-0.7462981243830207-blue)
![Average Recall](https://img.shields.io/badge/avg_recall-0.16886307795398706-blue)
![Average F1 Score](https://img.shields.io/badge/avg_f1-0.2754098360655738-blue)
![ROC AUC](https://img.shields.io/badge/roc_auc-0.5611651982201584-blue)
![Data Distribution Test](https://img.shields.io/badge/data_distribution-passing-brightgreen)

# Description

This is the ML training pipeline repository for group 6 for CS4295-Release Engineering for Machine Learning Applications.

# Pipeline design

The pipeline is divided into four stages:

1. Get data
2. Preprocess
3. Train
4. Test

Each of these stages has a respective file in the `src` folder. Furthermore, the `dvc.yaml` file describes the exact dependencies and outputs of these stages as well. 
We opted to divide the pipeline into these stages because each one has a well-defined input and output and follows standard ML practices.

# Prerequisites
- Python
        - Python `>= 3.11.*` or `< 3.12`.
- Poetry
        - Please, refer to the [official docs](https://python-poetry.org/docs/) for more information about Poetry. 
- DVC
        - Please, refer to the the [official docs](https://dvc.org/doc) for more informationa about DVC.

# Setup
- Create a new virtual environment called `env` using `virtualenv env`
- Activate the virtual environment you just created using `source <path-to-env>/bin/activate`
- Install Poetry using `pip install poetry`.
- Install dependencies using poetry by running `poetry install`.
- Sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (`https://www.kaggle.com/<username>/account`) and select 'Create API Token'. This will trigger the download of `kaggle.json`, a file containing your API credentials. Place this file in the location `~/.kaggle/kaggle.json`.

## Optional setup (for linters)

- Install Pylint using `pip install pylint`.
- Install Flake using `pip install flake8`.
- Install Autopep8 using `pip install autopep8`.

# Running instructions

To run the full data pipeline, use:

``` console
dvc exp run
```
Once the pipeline runs, you can see the results and the metrics by running
``` console
dvc exp show
```
To run pylint:

``` console
pylint <file/dir> > reports/output.txt
``` 
This should store the pylint output in `reports/output.txt`

To run autopep8:
``` console
autopep8 --in-place --aggressive --recursive <file/dir>
```

To run flake8:
``` console
flake8 <file/dir>
```

To run tests:
``` console
pytest tests/
```

### Model performance

[![DVC-metrics](https://img.shields.io/badge/dynamic/json?style=flat-square&colorA=grey&colorB=99ff99&label=Accuracy&url=https://raw.githubusercontent.com/remla24-team6/phishing_detection_cnn/main/reports/metrics.json&query=test_accuracy)](https://raw.githubusercontent.com/remla24-team6/phishing_detection_cnn/main/reports/metrics.json) 

[![DVC-metrics](https://img.shields.io/badge/dynamic/json?style=flat-square&colorA=grey&colorB=99ff99&label=Precision&url=https://raw.githubusercontent.com/remla24-team6/phishing_detection_cnn/main/reports/metrics.json&query=avg_precision)](https://raw.githubusercontent.com/remla24-team6/phishing_detection_cnn/main/reports/metrics.json) 

[![DVC-metrics](https://img.shields.io/badge/dynamic/json?style=flat-square&colorA=grey&colorB=99ff99&label=Recall&url=https://raw.githubusercontent.com/remla24-team6/phishing_detection_cnn/main/reports/metrics.json&query=avg_recall)](https://raw.githubusercontent.com/remla24-team6/phishing_detection_cnn/main/reports/metrics.json)

[![DVC-metrics](https://img.shields.io/badge/dynamic/json?style=flat-square&colorA=grey&colorB=99ff99&label=F-measure&url=https://raw.githubusercontent.com/remla24-team6/phishing_detection_cnn/main/reports/metrics.json&query=avg_f1)](https://raw.githubusercontent.com/remla24-team6/phishing_detection_cnn/main/reports/metrics.json) 

[![DVC-metrics](https://img.shields.io/badge/dynamic/json?style=flat-square&colorA=grey&colorB=99ff99&label=ROCAUC&url=https://raw.githubusercontent.com/remla24-team6/phishing_detection_cnn/main/reports/metrics.json&query=roc_auc)](https://raw.githubusercontent.com/remla24-team6/phishing_detection_cnn/main/reports/metrics.json) 

[![DVC-metrics](https://img.shields.io/badge/dynamic/json?style=flat-square&colorA=grey&colorB=99ff99&label=Inference-time&url=https://raw.githubusercontent.com/remla24-team6/phishing_detection_cnn/main/reports/metrics.json&query=inference_time)](https://raw.githubusercontent.com/remla24-team6/phishing_detection_cnn/main/reports/metrics.json) 





## Project Decisions

Below, we outline the decisions with regards to the set up and structure of our project.


### Remote
We configured a Google Drive folder as remote storage. It is accessible, [here](https://drive.google.com/drive/u/0/folders/1Bjf-9VYuLobnAYb6xvm_DwWWNlMuesrM).
To modify the remote in DVC, use:
```
> dvc remote add --default myremote \
                           gdrive://<gdrive-endpoint>
> dvc remote modify myremote gdrive_acknowledge_abuse true
> dvc push
```


### Project structure
We use the standard [Cookiecutter](https://drivendata.github.io/cookiecutter-data-science/) to ease the creation of directories and configuration files.

### Package management
We use [Poetry](https://python-poetry.org/) as our dependency management tool. Poetry ensures that users with different environments use the same versions of dependencies. 

### Linters
We use [Pylint](https://pypi.org/project/pylint/) to check for errors in our codebase, enforce a coding standard among our project, and look for bad code smells [^1] https://docs.pylint.org/intro.html#what-is-pylint.

Additionaly, we use flake8 as a linter as it also includes complexity analysis.

#### Pylint configuration
We use Pylint to check for code quality. The configuration file can be found in `.pylintrc`. 
Pylint was configured to accept variable names starting with a capital, such as X_train, X_test, because these are common ML variable names. 



```
Report
======
164 statements analysed.

Statistics by type
------------------

+---------+-------+-----------+-----------+------------+---------+
|type     |number |old number |difference |%documented |%badname |
+=========+=======+===========+===========+============+=========+
|module   |9      |9          |=          |100.00      |0.00     |
+---------+-------+-----------+-----------+------------+---------+
|class    |0      |NC         |NC         |0           |0        |
+---------+-------+-----------+-----------+------------+---------+
|method   |0      |NC         |NC         |0           |0        |
+---------+-------+-----------+-----------+------------+---------+
|function |11     |11         |=          |100.00      |0.00     |
+---------+-------+-----------+-----------+------------+---------+



External dependencies
---------------------
::

    kaggle
      \-api
        \-kaggle_api_extended (src.data.make_dataset)
    ml_lib_remla
      \-preprocessing (src.build_features)
    model (src.train)
    numpy (src.test)
    sklearn
      \-metrics (src.test)
    yaml (src.utils.io)



384 lines have been analyzed

Raw metrics
-----------

+----------+-------+------+---------+-----------+
|type      |number |%     |previous |difference |
+==========+=======+======+=========+===========+
|code      |205    |53.39 |205      |=          |
+----------+-------+------+---------+-----------+
|docstring |94     |24.48 |94       |=          |
+----------+-------+------+---------+-----------+
|comment   |1      |0.26  |1        |=          |
+----------+-------+------+---------+-----------+
|empty     |84     |21.88 |84       |=          |
+----------+-------+------+---------+-----------+



Duplication
-----------

+-------------------------+------+---------+-----------+
|                         |now   |previous |difference |
+=========================+======+=========+===========+
|nb duplicated lines      |0     |0        |0          |
+-------------------------+------+---------+-----------+
|percent duplicated lines |0.000 |0.000    |=          |
+-------------------------+------+---------+-----------+



Messages by category
--------------------

+-----------+-------+---------+-----------+
|type       |number |previous |difference |
+===========+=======+=========+===========+
|convention |0      |0        |0          |
+-----------+-------+---------+-----------+
|refactor   |0      |0        |0          |
+-----------+-------+---------+-----------+
|warning    |0      |0        |0          |
+-----------+-------+---------+-----------+
|error      |0      |0        |0          |
+-----------+-------+---------+-----------+



Messages
--------

+-----------+------------+
|message id |occurrences |
+===========+============+




--------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)
```
# Testing Design
We implemented the following tests:

- Test data distribution: this test makes sure the training data has a reasonable proportion of legitimate and phishing URLs. This is important as unbalanced data can result in less correct prediction models.
- Test data slice: Tests that the model predicts correctly for URLs containing `.com` and `.uk`. We test this to guarantee that the model has correct inference behavior on all kinds of URLs. 
- Test model nondeterminism: Tests that train the model on different random seeds result in similar accuracy scores. Testing correctness in the face of non-determinism is tested as it makes our results predictable.
- Test model-determinism: Tests that train the model on the same seed results in similar accuracy scores. We test model determinism to make sure the models are reliable and have predictable behavior.
- Test reproducibility: Ensures that the ML infrastructure is correct. 
- Test inference: Tests if the trained model has acceptable serving latency. The inference time must be short because otherwise, our use-case is of significantly less value. 
- Test memory: Tests that the memory usage does not exceed the set threshold. This test makes sure we make effective use of scarce memory resources.
- Test mutamorphic properties: Tests that trained model return the same result for original and mutated URLs from the test set. Applies automatic inconsistency repair by training the model on failed mutated URLs. Our mutamorphic tests check for robustness to different URLs and repairs the model where necessary to ensure high model correctness.

All of these tests can be found in the `tests/` folder. They are triggered through the DVC pipeline.
