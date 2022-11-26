import pandas as pd
import os
import numpy as np
import evaluate

from os.path import join
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding, PretrainedConfig
from classification_utils import preprocess, extract_value, map_labels
from datasets import ClassLabel

# define training parameters
path_df = join(".", "data", "classification", "df.csv")
x_name, y_name = "x_reduce", "y"
num_epochs = 10
cuda_device = "2"
model_name, batch_size, model_max_length = "camembert-base", 20, 512  # bert-base-multilingual-cased ; camembert-base
eval_batch_size = 128
dev = True
output_dir = "./results/full/bert/"
output_best_model = join(output_dir, str(num_epochs) + "_epochs")

# load data
df = pd.read_csv(path_df, sep="\t", index_col=0)

# encode y
# multi_label_bin = MultiLabelBinarizer()
# multi_label_bin.fit(df[y_name])
# y = multi_label_bin.transform(df[y_name])

# load labels
y_value = extract_value(df, y_name=y_name)

# load metrics
accuracy = evaluate.load("accuracy")

# split train dev test
if dev:
    X, X_test, Y, y_test = train_test_split(df[x_name], y_value, test_size=0.15)
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.06)  # 80 ; 5 ; 15
else:
    X_train, X_test, y_train, y_test = train_test_split(df[x_name], y_value, test_size=0.2, train_size=0.8)

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
label_name = "y"
labels = ClassLabel(names=list(set(y_value)))

# If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
# Otherwise, we have to get the list of labels manually.
assert isinstance(labels, ClassLabel)
label_list = labels.names
label_to_id = {i: i for i in range(len(label_list))}
assert len(label_list) == len(label_to_id)

# load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_list))

# Set the correspondences label/ID inside the model config
model.config.label2id = {l: i for i, l in enumerate(label_list)}
model.config.id2label = {i: l for i, l in enumerate(label_list)}

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=model_max_length)

print("Config is:")
print("train size:", X_train.shape[0], "; dev size:", X_val.shape[0], "; test size:", X_test.shape[0])
print("label number:", df[y_name].nunique())
print("model:", model_name)
print("epochs:", num_epochs)
print("batch_size:", batch_size)
print("model_max_length:", model_max_length)
print("model will be write in", output_best_model, "folder.")

# preprocess data
if dev:
    dev_dataset = preprocess(pd.DataFrame({"text": X_val.tolist(), "y": y_val}), "text", label_name, labels, tokenizer)

train_dataset = preprocess(pd.DataFrame({"text": X_train.tolist(), "y": y_train}), "text", label_name, labels,
                           tokenizer)
test_dataset = preprocess(pd.DataFrame({"text": X_test.tolist(), "y": y_test}), "text", label_name, labels, tokenizer)

# training arguments
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    save_strategy="no",
    load_best_model_at_end=True,
    do_train=True,
    do_eval=dev
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    score = accuracy.compute(predictions=predictions, references=labels)
    return score


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# train
trainer.train()
trainer.save_model(output_best_model)

# evaluation
eval_arguments = TrainingArguments(
    output_dir="empty",
    do_train=False,
    do_eval=True,
    per_device_eval_batch_size=eval_batch_size,
    dataloader_drop_last=False
)


def compute_metrics(eval_predictions):
    logits, label_ids = eval_predictions
    if isinstance(logits, tuple):
        logits = logits[0]  # [0] is logits
    predictions = np.argmax(logits, axis=-1)

    score = {}
    score["accuracy"] = accuracy.compute(references=label_ids, predictions=predictions)["accuracy"]

    print(score)
    return score


evaluator = Trainer(
    model=model,
    args=eval_arguments,
    tokenizer=tokenizer,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# evaluate
evaluator.evaluate()
