from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from classification_utils import add_prefix_to_x
import torch
import pandas as pd


df = pd.read_csv("./data/classification/df.csv", sep="\t", index_col=0)
df = add_prefix_to_x(df, prefix="multilabel classification: ", x_name="x", initial_x_name="x_full")

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

inputs = df["x"].tolist()
outputs = df["y_dictionnary"].tolist()

print("input lengths:", len(inputs))
print("output lengths:", len(outputs))

model_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt", max_length=1024)
with tokenizer.as_target_tokenizer():
    labels = tokenizer(outputs, padding=True, truncation=True, return_tensors="pt", max_length=1024)
model_inputs["labels"] = labels["input_ids"]

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.examples.items()}

    def __len__(self):
        return len(self.examples["input_ids"])

train = MyDataset(model_inputs)

training_args = Seq2SeqTrainingArguments(
    output_dir="output-t5-finetuned",
    overwrite_output_dir=True,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    run_name="T5 Experiment",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train,
    tokenizer=tokenizer
)

trainer.train()

