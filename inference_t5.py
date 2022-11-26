import pandas as pd
import os
import torch

from os.path import join
from tqdm import tqdm
from datasets import ClassLabel
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from classification_utils import add_prefix_to_x, preprocess_t5, t5_dataset, extract_value

# define arguments
path_df = join(".", "data", "classification", "df.csv")
path_pretrained_model = join(".", "results", "t5", "base-t5", "10_epochs")
cuda_device = "0"
eval_batch_size = 64
x_name="x_full"

# load dataset
df = pd.read_csv(path_df, sep="\t", index_col=0)

# add prefix
df = add_prefix_to_x(df, prefix="multilabel classification: ", x_name="x", initial_x_name=x_name)
x_name = "x"

# adapt y
y = extract_value(df, transition_word=", ")
class_label = ClassLabel(names=list(set(y)))

# select GPU(s)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

# load fine-tuned model
tokenizer = MT5Tokenizer.from_pretrained(path_pretrained_model)
model = MT5ForConditionalGeneration.from_pretrained(path_pretrained_model)

# preprocess text
preprocess_text = preprocess_t5(df["x"].tolist(), y, tokenizer)
dataset = t5_dataset(preprocess_text)

# predict
predict_arguments = Seq2SeqTrainingArguments(
    output_dir="empty",
    do_train=False,
    do_eval=False,
    do_predict=True,
    per_device_eval_batch_size=eval_batch_size,
    dataloader_drop_last=False,
    predict_with_generate=True
)

# prediction arguments
predictor = Seq2SeqTrainer(
    model=model,
    args=predict_arguments,
    tokenizer=tokenizer
)

# predict
predictions = predictor.predict(dataset)
def decode_predictions(predictions):
    generated_tokens_ids, label_ids, _ = predictions
    if isinstance(generated_tokens_ids, tuple):
        generated_tokens_ids = generated_tokens_ids[0]
    decoded_preds = tokenizer.batch_decode(generated_tokens_ids, skip_special_tokens=True)
    return decoded_preds

text_predictions = decode_predictions(predictions)
df["predictions"] = text_predictions
df.to_csv(join(".", "data", "classification", "df_with_predictions.csv"), sep="\t")

