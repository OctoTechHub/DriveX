import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load a dataset
df = pd.read_csv('models/file_data.csv') 
# Tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Preprocessing function to tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# DataLoader
train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(500))  # Select subset for quick training
test_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(500))

train_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./models/distilbert-seq-class')
tokenizer.save_pretrained('./models/distilbert-seq-class')
