# MLOPS Review Classifier

Dataset used to train classifier: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

This project is a Review Classifier built using a distilBERT model. The model was trained to analyze the sentiment of reviews and classify them as 'good', 'neutral', or 'bad'.

## Features

Fine-tuned DistilBERT model for review classification.
Dockerized application for easy deployment and scaling.
Deployed on Kubernetes for production-grade deployment.
RESTful API to interact with the model for inference.


## How to Use

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/amirkadriju/mlops-review-classifier.git
    ```

2. **Navigate to the Directory**:
    ```bash
    cd DataMastersRepo
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements-dev.txt
    ```

## Model Training
The model can be fine-tuned using the dataset above. 
1. Run the data_classification.py file to label the reviews
2. The labeled data can then be used to fine-tune the model using train.py
3. After training predict.py can be used to check the classification performance.


## Build Docker Setup
1. **Build Docker Image**:
    ```bash
    docker build -t review-classifier .
    ```

2. **Run Docker Container**:
    ```bash
    docker run -p 5000:5000 review-classifier
    ```

## Kubernetes Setup
1. **Create Docker Image and push to registry**:
    ```bash
    docker tag review-classifier yourdockerhubusername/review-classifier:latest
    docker push yourdockerhubusername/review-classifier:latest
    ```

2. **Apply Kubernetes Configurations using the two files in the k8s folder**:
    ```bash
    kubectl apply -f k8s/
    ```

