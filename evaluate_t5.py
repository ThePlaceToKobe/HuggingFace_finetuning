import pandas as pd
import numpy as np
import os
import evaluate

from os.path import join
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer, MT5ForConditionalGeneration
from sklearn.model_selection import train_test_split
from classification_utils import add_prefix_to_x, extract_value, preprocess_t5, t5_dataset
from datasets import ClassLabel

path_pretrained_model = join(".", "results", "full", "t5", "large-t5", "5_epochs")
path_data = join(".", "data", "classification", "df.csv")
x_name, y_name = "x_reduce", "y"
cuda_device = "2"
eval_batch_size = 100

# load data
df = pd.read_csv(path_data, sep="\t", index_col=0)

# adapt x
df = add_prefix_to_x(df, prefix="multilabel classification: ", x_name="x", initial_x_name=x_name)
x_name = "x"

# adapt y
y = extract_value(df, transition_word=", ", y_name=y_name)
class_label = ClassLabel(names=list(set(y)))

# split train dev test
X, X_test, y, y_test = train_test_split(df[x_name], y, train_size=0.8)

# select GPU(s)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

# load pretrained model
tokenizer = T5Tokenizer.from_pretrained(path_pretrained_model)
model = MT5ForConditionalGeneration.from_pretrained(path_pretrained_model)

# preprocess test dataset
preprocess_test = preprocess_t5(X_test.tolist(), y_test, tokenizer)
test_dataset = t5_dataset(preprocess_test)

# testing arguments
eval_arguments = Seq2SeqTrainingArguments(
    output_dir="empty",
    do_train = False,
    do_eval = True,
    per_device_eval_batch_size = eval_batch_size,
    dataloader_drop_last = False,
    predict_with_generate=True
)

bleu = evaluate.load("sacrebleu")
accuracy = evaluate.load("accuracy")

def post_process_bleu(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def post_process_accuracy(decoded_, class_label):
    encoded = []
    for i, l in enumerate(decoded_):
        if l in class_label.names:
            encoded.append(class_label.str2int(l))
        else:
            # predictions is not in class_label names so it is incorrect
            # We could investigate these predictions, maybe there are not far from a good one.
            encoded.append(-1)
    return encoded

def compute_metrics(eval_predictions):
    # print(eval_predictions)
    generated_tokens_ids, label_ids = eval_predictions
    if isinstance(generated_tokens_ids, tuple):
        generated_tokens_ids = generated_tokens_ids[0]  # [0] is logits
#    generated_tokens_ids = np.argmax(logits, axis=-1)
    decoded_preds = tokenizer.batch_decode(generated_tokens_ids, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    score = {}
    # compute accuracy
    predictions = post_process_accuracy(decoded_preds, class_label)
    labels = post_process_accuracy(decoded_labels, class_label)
    score["accuracy"] = accuracy.compute(references=labels, predictions=predictions)["accuracy"]
    
    # compute bleu score
    predictions, labels = post_process_bleu(decoded_preds, decoded_labels)
    score["bleu"] = bleu.compute(references=labels, predictions=predictions)["score"]

    print(score)
    return score

# evaluation arguments
evaluator = Seq2SeqTrainer(
    model = model,
    args = eval_arguments,
    tokenizer=tokenizer,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# evaluate
evaluator.evaluate()

