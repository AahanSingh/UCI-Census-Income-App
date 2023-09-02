# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
[Aahan Singh](https://github.com/AahanSingh) trained this model using the [UCI Census Income Dataset](https://archive.ics.uci.edu/dataset/20/census+income).
- Model Trained: **Random forest classifier**
    - Max depth: **50**
    - Number of estimators: **500**
- Latest version trained on **2/9/2023**
- Model synced with DVC. To get latest model run `dvc pull`. Model along with the encoder is stored under `./model/` directory.

## Intended Use
This model should be used to predict the salary range of a user given a set of parameters about the user. 

## Training Data

The model was trained on the [UCI Census Income Dataset](https://archive.ics.uci.edu/dataset/20/census+income). 

As part of the cleaning procedure, white spaces were removed from the csv file and saved at `data/census-clean.csv`. The data has been pushed to the data store via DVC. To obtain the data run `dvc pull`.
A random 80/20 split of the cleaned dataset was used as train and val datasets repectively.

## Evaluation Data

A random 80/20 split of the cleaned dataset was used as train and val datasets repectively.
The preprocessing steps involved a one hot encoding of the categorical columns.
The one hot encoding was used in the validation phase as well.


## Metrics
The following metrics were used to evaluate the model:
- Precision
- Recall
- F1 score (fbeta with beta set to 1)

The following is the model's overall performance on the validation set.
- Precision: 0.7349
- Recall: 0.6142
- F1: 0.6691

## Ethical Considerations

The Census Income dataset contains sensitive information about individuals, including their income, age, education, and occupation. To ensure ethical and legal use of the data, it is important to consider the following:
- The data was extracted from the 1994 Census database by Barry Becker. 
- The U.S. Census Bureau has privacy principles and data stewardship policies that guide their data collection practices and ensure confidentiality.
- Employees of the Census Bureau are held to a high standard of ethical conduct.
- Filling in missing data can pose ethical risks, particularly when it comes to race and ethnicity.
- Since the data is from 1991, it may not be wholly relevant in today's time. The predictions of the model should be taken with a grain of salt.

In summary, the Census Income dataset requires ethical considerations to protect the confidentiality of the information and ensure legal and ethical use of the data.

## Caveats and Recommendations

Due to the above ethical considerations, only a few categorical columns which did not have missing values were used.
The following columns were utilised in the training of the model for the prediction of salaries:
- workclass
- education
- marital_status
- occupation
- relationship
- race
- sex
- native_country

If we were to get more recent data, the model might perform better. 
Additionally, only random forests were tested in this experiment. Other methods such as boosting might provide better results. 

Other recommendations include
- Removal of rows with **?** values from the following columns can be considered outliers and removed
    - workclass
    - occupation
    - native_country
- Normalization of continuous features
