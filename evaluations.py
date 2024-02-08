import pickle

import pandas as pd


# class EventDurationPredictionDataset(object):
#     def __init__(self):
#         self.df = pd.read_csv("./event_duration_prediction_dataset.csv")
#         self.train_df = self.df.loc[self.df["event_type"].isin(["Hurricane Florence 2018", "Hurricane Sally 2020"])]
#         self.valid_df = self.df.loc[self.df["event_type"].isin(["Hurricane Laura 2020", "Saddleridge Wildfire 2019"])]
#         self.test_df = self.df.loc[self.df["event_type"].isin(["2018 Maryland Flood", "Lilac Wildfire 2017", "Cranston Wildfire 2018", "Holy Wildfire 2018"])]
#
#         self.train = self.train_df["text"]
#         self.val = self.valid_df["text"]
#         self.test = self.test_df["text"]
#
#         self.train_labels = self.train_df["regression_label"]
#         self.val_labels = self.valid_df["regression_label"]
#         self.test_labels = self.test_df["regression_label"]
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         return self.dataset[idx], self.labels[idx]


def get_sentence_indices_in_df(df, label_pkl, prediction_pkl, stratified_sample_indices_path, sentence_pairs_indices_path, output_path):
    stormy = True if "stormy_data" in sentence_pairs_indices_path else False
    with open(label_pkl, "rb") as fp:
        labels = pickle.load(fp)
    with open(prediction_pkl, "rb") as fp:
        predictions = pickle.load(fp)
    with open(stratified_sample_indices_path, "rb") as fp:
        stratified_sample_indices = pickle.load(fp)
    with open(sentence_pairs_indices_path, "rb") as fp:
        sentence_pairs_indices = pickle.load(fp)
    sentence_a = []
    sentence_b = []
    event_a = []
    event_b = []
    time_a = []
    time_b = []
    for sent_i in stratified_sample_indices:
        sent_pair_indices = sentence_pairs_indices[sent_i]
        if stormy:
            sentence_a.append(df.title.values[sent_pair_indices[0]])
            sentence_b.append(df.title.values[sent_pair_indices[1]])
            event_a.append(df.wikidata_link.values[sent_pair_indices[0]])
            event_b.append(df.wikidata_link.values[sent_pair_indices[1]])
            time_a.append(df.seendate.values[sent_pair_indices[0]])
            time_b.append(df.seendate.values[sent_pair_indices[1]])
        else:
            sentence_a.append(df.text.values[sent_pair_indices[0]])
            sentence_b.append(df.text.values[sent_pair_indices[1]])
            event_a.append(df.event.values[sent_pair_indices[0]])
            event_b.append(df.event.values[sent_pair_indices[1]])
            time_a.append(df.unix_timestamp.values[sent_pair_indices[0]])
            time_b.append(df.unix_timestamp.values[sent_pair_indices[1]])

    df = pd.DataFrame(list(zip(sentence_a, event_a, time_a, labels, predictions, sentence_b, event_b, time_b)),
                      columns =['sentence_a', 'event_a', 'time_a', 'labels', 'predictions', 'sentence_b', 'event_b', 'time_b'])
    df.to_csv(output_path)
    return df


if __name__ == "__main__":
    data_types = ["stormy_data", "crisisfacts_data"]
    tasks = ["event_deduplication", "event_temporality"]
    exp_names = ["v6"]
    for data_type in data_types:
        for task in tasks:
            for exp_name in exp_names:
                for mode in ["test"]:
                    # label_pkl = f"./outputs/{exp_name}/{task}/{mode}/{mode}_{exp_name}_{task}_labels.pkl"
                    # prediction_pkl = f"./outputs/{exp_name}/{task}/{mode}/{mode}_{exp_name}_{task}_prediction.pkl"
                    # stratified_sample_indices_path = f"./data/{data_type}/{task}/stratified_sample_indices_{mode}.pkl"
                    # sentence_pairs_indices_path = f"./data/{data_type}/{task}/sentence_pairs_indices_{mode}.pkl"
                    # df_path = f"./data/{data_type}/crisisfacts_{mode}.csv" if data_type == "crisisfacts_data" else f"./data/{data_type}/{mode}_v2.csv"
                    # df = pd.read_csv(df_path)
                    # output_path = f"./data/{data_type}/{task}/{mode}_df_{exp_name}.csv"
                    # df = get_sentence_indices_in_df(df, label_pkl, prediction_pkl, stratified_sample_indices_path,
                    #                                      sentence_pairs_indices_path, output_path)
                    # print(f"sentence pair df (len: {len(df)}) saved in {output_path}")
                    df_path = f"./data/{data_type}/{task}/{mode}.csv"
                    df = pd.read_csv(df_path)
                    print(f"sentence pair df len: {len(df)})")
                    print(f"unique sentence_a len: {len(df.sentence_a.value_counts())})")
                    print(f"unique sentence_b len: {len(df.sentence_b.value_counts())})")
                    print(f"unique event_a len: {len(df.event_a.value_counts())})")
                    print(f"unique event_b len: {len(df.event_b.value_counts())})")
                    print(f"unique time_a len: {len(df.time_a.value_counts())})")
                    print(f"unique time_b len: {len(df.time_b.value_counts())})")
                    print(f"unique labels len: {len(df.labels.value_counts())})")
                    print("")

