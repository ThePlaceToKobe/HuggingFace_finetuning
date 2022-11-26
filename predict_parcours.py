from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TextClassificationPipeline
from utils_inference import *

# define arguments
path_df = join(".", "data", "toc_msan_2022.csv")
path_pretrained_model = join(".", "results", "full", "bert", "10_epochs")
cuda_device = "0"
x_name = "x_reduce"

# load dataset
df = pd.read_csv(path_df, sep="\t", encoding="ISO-8859-1", low_memory=False)
df_ticket = preprocess_parcours(df)
print(df_ticket.ticket_id.nunique(), "dialogs.")
x_name = "x"

# load fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(path_pretrained_model)
model = AutoModelForSequenceClassification.from_pretrained(path_pretrained_model)

# creation prediction pipeline
prediction_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=3)

# df_tmp = df_ticket[:1000]

predictions = []
for x in tqdm(df_ticket["x"]):
    if "no_label" in x:
        prediction = "no_comment"
        score = 1
    else:
        predictions.append(prediction_pipeline(x)[0][0])

df_ticket.keys()
from utils import create_y_t5
vars = ["action_reset", "action_changement", "action_cablage", "action_états"]
files = ['--_directions_eds_--.txt', '--_identification_infos_de_reset_--.txt',
         '--_identification_infos_de_changements_--.txt',
         '--_identification_infos_cablage_et_trans_--.txt', '--_infos_sur_différents_états_--.txt',
         '--_identification_infos_autres_--.txt']
path_rule_folder=join(".", "data", "python_rules", "txt")
for name_r, f in zip(vars, files):
    print("apply rule", name_r)
    df_ticket = apply_rule(df_ticket=df_ticket, var_name=name_r, path_rule=join(path_rule_folder, f))

df_ticket = create_y_t5(df_ticket, vars, var_y_name="y")



# to send
df_ticket.keys()
df_ticket = df_ticket.drop(["comment_origincomment", "comment_maincomment", "action_reset", "action_changement", "action_cablage",
            "action_états"], axis=1)


df_ticket.to_csv(join(".", "data", "parcours"))


from collections import Counter

Counter(df_tmp.y)
Counter([p["label"] for p in predictions])

# write predictions
x_to_predict = pd.concat((df_ticket, predictions), axis=1)
x_to_predict.to_csv(join(".", "data", "classification", "df_predictions.csv"))
