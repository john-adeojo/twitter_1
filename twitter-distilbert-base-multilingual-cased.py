import sys 
sys.path.append('/home/data-scientist/anaconda3/envs/twitter-env-1/lib/python3.9/site-packages')
import pandas as pd
import numpy as np

df = pd.read_csv('/home/data-scientist/anaconda3/envs/twitter-env-1/data/raw data/twitter_data.csv')

class DataProcessor:
    
    def __init__(self, df):
        self.df = df.copy()
    
    def drop_unknown_users(self):
        self.df = self.df.dropna(subset=['gender'])
        self.df = self.df[self.df['gender'] != 'unknown']
    
    def replace_blank_user_description(self):
        self.df['user_description'] = self.df['user_description'].fillna('no user description')

        
    def drop_columns(self, columns_to_drop):
        self.df = self.df.drop(columns=columns_to_drop)
        
    def isolate_duplicates(self):
        duplicate_names = self.df[self.df.duplicated('name')]['name']
        self.duplicates = self.df[self.df['name'].isin(duplicate_names)]
        self.df = self.df[~self.df['name'].isin(duplicate_names)]
    
    def split_data(self):
        # Split into train, validation, and test sets
        train_pct = 0.7
        val_pct = 0.15
        test_pct = 0.15

        # Set random seed for reproducibility
        np.random.seed(42)

        # Shuffle the data
        shuffled_df = self.df.sample(frac=1)

        # Determine the number of rows for each set
        num_rows = len(shuffled_df)
        train_rows = int(num_rows * train_pct)
        val_rows = int(num_rows * val_pct)
        test_rows = num_rows - train_rows - val_rows

        # Split the data into sets
        self.train_df = shuffled_df.iloc[:train_rows]
        self.val_df = shuffled_df.iloc[train_rows:train_rows+val_rows]
        self.test_df = shuffled_df.iloc[train_rows+val_rows:]
        
        # Add duplicates to training set
        self.train_df = pd.concat([self.train_df, self.duplicates])
        
        # Ensure no overlap in names
        train_names = set(self.train_df['name'])
        val_names = set(self.val_df['name'])
        test_names = set(self.test_df['name'])
        
        assert len(train_names.intersection(val_names)) == 0, "Overlap in names between train and validation sets"
        assert len(train_names.intersection(test_names)) == 0, "Overlap in names between train and test sets"
        assert len(val_names.intersection(test_names)) == 0, "Overlap in names between validation and test sets"
        
    def process_data(self, cols_to_drop):
        
        self.drop_unknown_users()
        self.replace_blank_user_description()
        self.drop_columns(cols_to_drop)
        self.isolate_duplicates()
        self.split_data()


processor = DataProcessor(df)
cols_to_drop = ["tweet_coord", "user_timezone", "tweet_location", "unit_id"]
processor.process_data(cols_to_drop)
# create train, validation and test
train_df = processor.train_df
validation_df = processor.val_df
test_df = processor.test_df


# Create labels and features

def features_labels(df, label, feature):
    y = list(df[label].astype("category").cat.codes)
    X = list((df[feature]))
    return y, X

train_labels, train_user_description = features_labels(train_df, label="gender", feature="user_description")
validation_labels, validation_user_description = features_labels(validation_df, label="gender", feature="user_description")
test_labels, test_user_description = features_labels(test_df, label="gender", feature="user_description")


# Create Dataset object

import torch

class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments

MODEL ='distilbert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)

train_encodings = tokenizer(train_user_description, truncation=True, padding=True)
val_encodings = tokenizer(validation_user_description, truncation=True, padding=True)
test_encodings = tokenizer(test_user_description, truncation=True, padding=True)

train_dataset = TwitterDataset(train_encodings, train_labels)
validation_dataset = TwitterDataset(val_encodings, validation_labels)
test_dataset = TwitterDataset(test_encodings, test_labels)


from transformers import AutoModelForSequenceClassification
training_args = TrainingArguments(
    output_dir='/home/data-scientist/anaconda3/envs/twitter-env-1/data/model training results/',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='/home/data-scientist/anaconda3/envs/twitter-env-1/model training logs/',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy='steps',
    load_best_model_at_end=True,
    
)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)

trainer = Trainer(
    model=model,                         # the instantiated  Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=validation_dataset      # evaluation dataset                              
    
)

trainer.train()


trainer.save_model("/home/data-scientist/anaconda3/envs/twitter-env-1/models/distilbert-base-multilingual-cased-best-model/")