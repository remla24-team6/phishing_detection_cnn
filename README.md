# Description

This is the final project for group 6 for CS4295-Release Engineering for Machine Learning Applications.

# Setup

- Create a virtual environment and run `pip install -r requirements.txt`. This should install the correct version for all dependencies and subdependencies.
- Sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (`https://www.kaggle.com/<username>/account`) and select 'Create API Token'. This will trigger the download of `kaggle.json`, a file containing your API credentials. Place this file in the location `~/.kaggle/kaggle.json`.

# Repository TODOs
- [X] Structure project
- [X] Add project requirements (might need to refactor this later)
- [X] Add notebook code in separate python modules so that they can be run as DVC stages.
- [ ] Set up DVC pipeline
- [ ] Set up remote DVC artefact repository.
- [ ] Set DVC metrics and experiment tracking.
- [ ] Set up pylint
- [ ] Set up Flake8
- [ ] Set up a static code formatter
- [ ] Update README with running instructions
- [ ] Document design decisions.


# Commands

To run pylint:
```
pylint <file/dir> > reports/output.txt
```

To run autopep8:
```
autopep8 --in-place --aggressive --recursive <file/dir>
```

To run flake8:
```
flake <file/dir>
```
