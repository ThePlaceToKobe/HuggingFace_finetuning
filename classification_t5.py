import pandas as pd
import os
import numpy as np
import evaluate

from torch import tensor
from os.path import join
from sklearn.model_selection import train_test_split
from transformers import MT5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from classification_utils import dict_to_ls_dict, extract_value, add_prefix_to_x, t5_dataset, preprocess_t5
from datasets import ClassLabel

# define training parameters
path_df = join(".", "data", "classification", "df.csv")
x_name, y_name = "x_reduce", "y"
num_epochs = 1
cuda_device = "2"

#model_name, batch_size, model_max_length = "google/mt5-small", 10, 1024  # 1.12G
#model_name, batch_size, model_max_length = "google/mt5-base", 3, 1024  # 2.17G
model_name, batch_size, model_max_length = "google/mt5-large", 1, 256  # 4.6G

output_dir = "./results/full/t5/" + model_name.split("-")[-1] + "-t5/"  # i.e. base-t5 
output_best_model = join(output_dir, str(num_epochs) + "_epochs")

# load dataset
df = pd.read_csv(path_df, sep="\t", index_col=0)

# adapt x
df = add_prefix_to_x(df, prefix="multilabel classification: ", x_name="x", initial_x_name=x_name)
x_name = "x"

# adapt y
y = extract_value(df, transition_word=", ", y_name=y_name)
class_label = ClassLabel(names=list(set(y)))

# select GPU(s)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

# load model
model = MT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=model_max_length)

print("Config is:")
print("label number:", df[y_name].nunique())
print("model:", model_name)
print("epochs:", num_epochs)
print("batch_size:", batch_size)
print("model_max_length:", model_max_length)
print("model will be write in", output_best_model, "folder.")

# split train test
dev = True

if dev:
    X, X_test, y, y_test = train_test_split(df[x_name], y, train_size=0.85)
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, train_size=0.94)

    preprocess_dev = preprocess_t5(X_dev.tolist(), y_dev, tokenizer)
    dev_dataset = t5_dataset(preprocess_dev)
    do_eval = True
else:
    X_train, X_test, y_train, y_test = train_test_split(df[x_name], y, train_size=0.8)
    do_eval = False

# preprocess text
preprocess_train = preprocess_t5(X_train.tolist(), y_train, tokenizer)
train_dataset = t5_dataset(preprocess_train)

# training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    num_train_epochs=num_epochs,
    do_eval=do_eval,
    load_best_model_at_end=True,
    save_strategy="no"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

# train
trainer.train()
trainer.save_model(output_best_model)

