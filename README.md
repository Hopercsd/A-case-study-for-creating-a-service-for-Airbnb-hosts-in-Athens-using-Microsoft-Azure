# README.md


## Project Overview

This project serves to build a model that predicts the prices of new Airbnb listings in Athens using previously listed accommodations. \
The trained model is deployed as a cloud service using Microsoft Azure. \
This project also explores undervalued listings based on the model as well as top-rated listings based on sentiment analysis on review data using Azure Cognitive Services.


## Directory and Files Structure
<pre>
|-- D03 
    |-- fastapi 
        |-- data 
            |-- regressor.joblib 
            |-- scaler.joblib 
        |-- src 
            |-- server.py 
        |-- Dockerfile 
        |-- check_api.ipynb 
        |-- requirements.txt 
    |-- undervalued-top_rated 
        |-- binary
            |-- reviews_most_recent.joblib
            |-- sentiment.joblib
        |-- identify.ipynb
        |-- text_analysis.ipynb
    |-- evaluation.py 
    |-- model.py
    |-- preprocessing.py
</pre>

## Dependencies 

requests==2.23.0 \
scikit-learn==1.0.1 \
pandas==1.3.0 \
numpy==1.21.0 \
fastapi==0.70.0 \
uvicorn==0.15.0 \
joblib==0.17.0 \
scipy==1.5.3 \
pydantic==1.8.2 \
googletrans==4.0.0-rc.1 \
bs4==4.10.0 \
azure-ai-textanalytics==5.1.0


## Run the Project

### Locally

Run `evaluation.py` which uses `preprocessing.py` in order to get the train and test data, evaluates different models and returns their respective metrics. \
Run `model.py` which uses `preprocessing.py` in order to get train and test data and stores the best regressor found in `evaluation.py` file in a .joblib file.

#### Undervalued and top-rated

Run `identify.ipynb` which finds undervalued and top-rated listings. It stores the most recent reviews in .joblib file to be used in the following file. \
Run `text_analysis.ipynb` file in order to obtain the positivity confidence score for the reviews of each listing.  

### FastAPI

Deployment of the model as a service by using fastapi web framework, Docker and Microsoft Azure. \
Get the IP Address and port number which are created in Azure. \
An example of a new case can be found in check_api.ipynb file. A new listing is passed and its price is predicted.
