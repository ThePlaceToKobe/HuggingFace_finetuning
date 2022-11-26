import os
import time
import pandas as pd

from transformers import pipeline
from os.path import join
from tqdm import tqdm
from classification_utils import extract_value, adjust_probability_predictions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss, accuracy_score
from classification_utils import hamming_score

# define parameters
path_df = join(".", "data", "classification", "df.csv")
x_name, y_name = "x_full", "y_dictionnary"
cuda_device = "2"

# model_name = "joeddav/xlm-roberta-large-xnli"
model_name = "cmarkea/distilcamembert-base-nli"  # 262M

# select GPU(s)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

# load data
df = pd.read_csv(path_df, sep="\t", index_col=0)

# load labels
candidate_labels = []
for v in df["y_dictionnary"]:
    for action in v.split(";"):
        y = eval(action).get(list(eval(action).keys())[0])
        if y not in candidate_labels:
            candidate_labels.append(y)

hypothesis_template = "Ce texte parle de {}."

classifier = pipeline('zero-shot-classification', model=model_name, tokenizer=model_name)

# define y
y_value = extract_value(df)

# split train,test
X_train, X_test, y_train, y_test = train_test_split(df[x_name], y_value, train_size=0.8)

#predictions = classifier(X_test.tolist(), candidate_labels, hypotheses_template=hypothesis_template)
# scores = [l for l in prediction["scores"] if l > 1 / 35]

start_time = time.time()
predictions = []
for i in tqdm(range(X_test.shape[0])):
    prediction = classifier(X_test.tolist()[i], candidate_labels, hypothesis_template=hypothesis_template)
    scores = [l for l in prediction["scores"] if l > 1 / 35]
    predictions.append((prediction["labels"][:len(scores)], scores))
end_time = time.time()

multi_label_bin = MultiLabelBinarizer()
multi_label_bin.fit(y_value)
labels = multi_label_bin.transform(y_test)

y_predictions = adjust_probability_predictions(predictions, probability=0.04)
y_predictions = multi_label_bin.transform(y_predictions)

print("Classifier:", model_name)
print("accuracy_score:", round(accuracy_score(labels, y_predictions), 2))  # accuracy_score: 0.79
print("Hamming_loss:", round(hamming_loss(labels, y_predictions), 2))  # Hamming_loss: 0.02
print("Hamming_score:", round(hamming_score(labels, y_predictions), 2))  # Hamming_score: 0.97
print("Learning_time:", "%s minutes" % round(((end_time - start_time) / 60), 2))  # Learning_time: 6.67 minutes
