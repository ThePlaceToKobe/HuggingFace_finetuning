import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join
from utils import apply_rule, clean_comment_label


def get_ticket_inference(df):
    ticket = {"ticket_id": [], "ticket_description": [], "ticket_shortlabel": [],
              "actor_group_label": [], "actor_name": [], "comment_id": [],
              "comment_recordingdate": [], "comment_label": [], "comment_origincomment": [],
              'comment_maincomment': []}

    for i in tqdm(range(df.shape[0])):
        com = eval(df.comments_comment.loc[i].replace("null", "'null'"))

        if com == "null":
            com = [com]

        for j, under_c in enumerate(com):
            if under_c != "null":
                actor_group_label = under_c["actor"]["group"]["label"]
                actor_name = under_c["actor"]["name"]
                recording_date = under_c["recordingdate"]
                label = under_c["label"]
                origin_comment = under_c["origincomment"]
                main_comment = under_c["maincomment"]
            else:
                actor_group_label = "no_actor_group_label"
                actor_name = "no_actor_name"
                recording_date = "no_recording_date"
                label = "no_label"
                origin_comment = "no_origin_comment"
                main_comment = "no_main_comment"

            # ticket
            ticket["ticket_id"].append(df.ticket_id.loc[i])
            ticket["ticket_description"].append(df.ticket_description.loc[i])
            ticket["ticket_shortlabel"].append(df.ticket_shortlabel.loc[i])

            ticket["actor_group_label"].append(actor_group_label)
            ticket["actor_name"].append(actor_name)

            # comments_comment
            ticket["comment_id"].append(j)
            ticket["comment_recordingdate"].append(recording_date)
            ticket["comment_label"].append(label)
            ticket["comment_origincomment"].append(origin_comment)
            ticket["comment_maincomment"].append(main_comment)

    df_ticket = pd.DataFrame(ticket)
    return df_ticket


def create_x(df_ticket):
    new_x = []
    for i in tqdm(range(df_ticket.shape[0])):
        x_tmp_reduce = "multilabel classification: "
        x_tmp_reduce += "comment label: " + df_ticket.loc[i, "comment_label"] + " "
        x_tmp_reduce += "directions eds: " + df_ticket.loc[i, "directions_eds"] + " "

        new_x.append(x_tmp_reduce)
    df_ticket["x"] = new_x
    return df_ticket


def concatenate_df_for_multi_index(df, index_whose_nb_index_sup_1, columns):
    df_multi = pd.DataFrame()
    for index in index_whose_nb_index_sup_1:
        df_tmp = df.loc[df.ticket_id == index]
        df_tmp = df_tmp[columns].sort_values(by=["tech_timestampchargement"]).reset_index(drop=True)
        index_max = np.argmax([len(df_tmp.loc[i, "comments_comment"]) for i in range(df_tmp.shape[0])])
        df_multi = df_multi.append(df_tmp.loc[index_max]).reset_index(drop=True)
    return df_multi


def preprocess_parcours(df, path_rule=join(".", "data", "python_rules", "txt", '--_directions_eds_--.txt')):
    # filter columns
    columns = ["tech_timestampchargement", "ticket_creationdate", "ticket_closuredate",
               "ticket_id", "ticket_description", "ticket_shortlabel", "ticket_actor_group_label",
               "contributor_activegroups_active", "contributor_initactor_actor_group_label",
               "contributor_pilotactor_actor_group_label", "comments_comment"]

    # filter unique ticket
    df_i, df_o = pd.DataFrame(df.ticket_id.value_counts() == 1), pd.DataFrame(df.ticket_id.value_counts() > 1)
    index_whose_nb_index_1 = [index for index in df_i.index if df_i.loc[index].values[0]]
    index_whose_nb_index_sup_1 = [index for index in df_o.index if df_o.loc[index].values[0]]
    df_unique = df.loc[df.ticket_id.isin(index_whose_nb_index_1)][columns]
    df_multi = concatenate_df_for_multi_index(df, index_whose_nb_index_sup_1, columns)
    df = pd.concat((df_multi, df_unique)).reset_index(drop=True)

    # filter comments_comment null
    # ticket_id_not_null = [df.loc[i].ticket_id for i in range(df.shape[0]) if not df.comments_comment.loc[i] == "(null)"]
    # df = df.loc[df.ticket_id.isin(ticket_id_not_null)].reset_index(drop=True)

    # df_ticket creation
    print("create ticket dataframes...")
    df_ticket = get_ticket_inference(df)

    # apply direction EDS rule
    print("get directions EDS...")
    df_ticket = apply_rule(df_ticket=df_ticket, var_name="directions_eds", path_rule=path_rule)

    # preprocess X
    print("clean comment...")
    df_ticket = clean_comment_label(df_ticket)

    # concatenate X and add prefix
    print("create X...")
    df_ticket = create_x(df_ticket)

    return df_ticket


def make_predictions(map_labels, prediction_pipeline, x):
    model_predictions = {"label": [], "score": []}
    for x in tqdm(x):
        prediction, score = "", -1
        if "no_label" in x:
            prediction = "no_comment"
            score = 1
        else:
            predictions = prediction_pipeline(x)

            for p in predictions[0]:
                label = map_labels.loc[map_labels.num_class == p["label"], "label"].tolist()[0]
                if label == "autres":
                    continue
                else:
                    prediction = label
                    score = p["score"]
                    break

        model_predictions["label"].append(prediction)
        model_predictions["score"].append(score)
    return pd.DataFrame(model_predictions)


def add_comment_id(df_ticket):
    comment_id, cpt = [], 0
    for i in tqdm(range(df_ticket.shape[0])):
        if i == 0:
            old_id = t_id = df_ticket.ticket_id.iloc[i]
        else:
            old_id, t_id = df_ticket.ticket_id.iloc[i-1], df_ticket.ticket_id.iloc[i]
        if old_id == t_id:
            comment_id.append(cpt)
        else:
            cpt = 0
            comment_id.append(cpt)
        cpt += 1
    df_ticket["comment_id"] = comment_id
    return df_ticket
