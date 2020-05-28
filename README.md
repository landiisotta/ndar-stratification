# ASD adaptive behavior subtypes from NDAR dataset

We present here how to run the code to enable the stratification of Vineland scores from the NDAR database.

### Requirements
```
Python >3.6
``` 

Install libraries with:
```
pip install -r requirements.txt
```

### `utils.py` file
Create an utils file including variables as reported in `example_utils.py`

### Pipeline
1. Once the `utils.py` file is ready, run `dataset` method from `create_dataset` to obtain 
the long-format Vineland dataset and the dictionary of demographic information for each
selected subject saved as `namedtuple`.

    Preprocessing steps include:

    > Phenotype selection (`import_data:_select_phenotype`);

    > Read tables from multiple instrument 
    >datasets (`import_data:ReadData:data_wrangling`);

    > Merging of different instrument versions 
    >(e.g., renaming feature names) with `create_levels:vineland_levels`; 

2. Run the `create_dataset:prepare_imputation` function that drops subjects that have 
a percentage of missing information greater than the desired threshold.

3. Grid search the best parameters for the Relative Clustering Validation. Considered
different missing percentages thresholds and varying number of neighbors.