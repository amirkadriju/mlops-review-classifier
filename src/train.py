import torch
import evaluate
import mlflow
import mlflow.pytorch
import mlflow.transformers
import pandas as pd
from datetime import datetime
from datasets import Dataset, DatasetDict
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# loads dataset --> nr samples is selected, because training on full dataset would be very time consuming
def load_dataset(path, nr_samples_per_class=100):
    # load dataset with labels
    df = pd.read_csv(path)

    # select only columns we need for classification task
    df = df[['Text', 'labels']]

    # get same number of data across all samples
    df = (df.groupby('labels')
        .apply(lambda x: x.sample(n=nr_samples_per_class, random_state=42))
        .reset_index(drop=True))
    
    return df


# reads unique labels & creates maps of label to id & id to label
# also df gets transformed into huggingface dataset
def get_huggingface_dataset_and_label_maps(df):
    # get unique labels from entire dataset
    unique_labels = sorted(set(df['labels']))

    # create maps of label 2 id, id 2 label and list of all numeric labels
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    numeric_labels = [label2id[label] for label in unique_labels]

    # label mapping --> get numeric labels into dataframe
    df['labels'] = df['labels'].map(label2id)

    # convert my df to a huggingface dataset
    dataset = Dataset.from_pandas(df)

    return dataset, unique_labels, label2id, id2label, numeric_labels


# load tokeniker
def load_tokenizer():
    return DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


# Tokenization function
def tokenize_function(example):
    return tokenizer(example['Text'], truncation=True, padding='max_length', max_length=300)


# splits data into train and test
def get_train_test_split(tokenized_dataset):
    # Split dataset into train and test
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = tokenized_dataset['train']
    eval_dataset = tokenized_dataset['test']

    return train_dataset, eval_dataset


# metrics for training evaluation
def compute_metrics(eval_pred):
    # define evaluation metric for training
    accuracy_metric = evaluate.load('accuracy')
    
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


if __name__ == '__main__':
    mlflow.set_experiment('DistilBERT_Review_Classifier')
    # set parameters
    nr_samples_per_class = 3000
    lr = 2e-5
    epochs = 3
    batch_size = 32
    weight_decay = 0.01
    
    run_name = f'run_lr{lr}_wd{weight_decay}_ep{epochs}_bs{batch_size}_samples{nr_samples_per_class}'
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with mlflow.start_run(run_name=run_name):
        # set mlflow tags
        mlflow.set_tag('version', 'v1')
        mlflow.set_tag('notes', 'First test run with small sample size')
        
        # Log training parameters manually
        mlflow.log_param('model', 'distilbert-base-uncased')
        mlflow.log_param('learning_rate', lr)
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('weight_decay', weight_decay)
        mlflow.log_param('nr_samples_per_class', nr_samples_per_class)
        
        # load dataset
        path = './data/reviews_with_labels.csv'
        df_raw = load_dataset(path, nr_samples_per_class)

        # make df ready for tokenization
        dataset, unique_labels, label2id, id2label, numeric_labels = get_huggingface_dataset_and_label_maps(df_raw)

        # load tokenizer
        tokenizer = load_tokenizer()

        # tokenize dataset & Format the dataset for PyTorch
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        # split data into train and eval
        train_dataset, eval_dataset = get_train_test_split(tokenized_dataset)

        # load distilbert for classification
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=len(unique_labels),
            id2label=id2label,
            label2id=label2id
        )

        #  set training arguments
        training_args = TrainingArguments(
            output_dir='./distilbert-review-classifier',
            eval_strategy='epoch',
            save_strategy='epoch',
            logging_strategy='epoch',
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy'
        )

        # define trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer
        )

        # start training process
        trainer.train()

        # Save model locally and log it with MLflow
        save_path = f'./checkpoints/{timestamp}_{run_name}'
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)

        # Log the entire model with mlflow
        mlflow.pytorch.log_model(trainer.model, artifact_path='model')
        mlflow.log_artifacts(save_path, artifact_path='tokenizer')

        # log final accuracy
        metrics = trainer.evaluate()
        mlflow.log_metric('eval_accuracy', metrics['eval_accuracy'])

    # Get predictions
    predictions = trainer.predict(eval_dataset)

    # Extract labels and predicted classes
    y_true = predictions.label_ids
    y_pred = predictions.predictions.argmax(-1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Optional: label names if you have id2label mapping
    labels = list(id2label.values())  # or just ["negative", "positive"] etc.

    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
