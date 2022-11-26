import pandas as pd
import time

from os.path import join
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, accuracy_score, f1_score
from classification_utils import hamming_score

df = pd.read_csv(join(".", "data", "classification", "df.csv"), sep="\t", index_col=0)

# Encode y
multi_label_bin = MultiLabelBinarizer()
multi_label_bin.fit(df["y"])
y = multi_label_bin.transform(df["y"])

# Encode X
count_v = CountVectorizer()
X_counts = count_v.fit_transform(df["x_reduce"])

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# split train test
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=824)

# config
print("train size:", X_train.shape[0], "test size:", X_test.shape[0])
print("model: RandomForestClassifier")
print("label size:", df["y"].nunique())

# fit model
start_time = time.time()
random_forest = RandomForestClassifier()
clf = OneVsRestClassifier(random_forest)
clf.fit(X_train, y_train)
y_predictions = clf.predict(X_test)
end_time = time.time()
print("--- %s minutes ---" % round(((end_time - start_time) / 60), 2))

print("Classifier:", clf)
print("accuracy_score:", round(accuracy_score(y_test, y_predictions), 2))  # accuracy_score: 0.79
print("Hamming_loss:", round(hamming_loss(y_test, y_predictions), 2))  # Hamming_loss: 0.02
print("Hamming_score:", round(hamming_score(y_test, y_predictions), 2))  # Hamming_score: 0.97
print("F1_score:", round(f1_score(y_test, y_predictions, average="macro"), 2))  # Hamming_score: 0.97
print("Learning_time:", "%s minutes" % round(((end_time - start_time) / 60), 2))  # Learning_time: 6.67 minutes
