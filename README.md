# Description

This is the data pipeline repository for group 6 for CS4295-Release Engineering for Machine Learning Applications.

# Pipeline design

The pipeline is divided into four stages:

1. Get data
2. Preprocess
3. Train
4. Test

Each of these stages has a respective file in the src folder. Furthermore, the `dvc.yaml` file describes the exact dependencies and outputs of these stages as well. 
We opted to divide the pipeline into these stages because each one has a well-defined input and output and follows standard ML practices.

# Setup
- Create a new virtual environment called `env` using `virtualenv env`
- Activate the virtual environment you just created using `source <path-to-env>/bin/activate`
- Install Poetry using `pip install poetry`.
- Install dependencies using poetry by running `poetry install`.
- Sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (`https://www.kaggle.com/<username>/account`) and select 'Create API Token'. This will trigger the download of `kaggle.json`, a file containing your API credentials. Place this file in the location `~/.kaggle/kaggle.json`.

## Optional setup (for linters)

- Install Pylint using `pip install pylint`
- Install Flake using `pip install flake8`

# Running instructions

To run the full data pipeline, use:

```
dvc exp run
```
Once the pipeline runs, you can see the results and the metrics by running
```
dvc exp show
```
To run pylint:

```
pylint <file/dir> > reports/output.txt
```
This should store the pylint output in `reports/output.txt`

To run autopep8:
```
autopep8 --in-place --aggressive --recursive <file/dir>
```

To run flake8:
```
flake8 <file/dir>
```

The 


