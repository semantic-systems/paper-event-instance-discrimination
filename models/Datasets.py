from math import floor
import logging
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
import random
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def split_stormy_dataset():
    df = pd.read_csv("./data/stormy_data/final_df_v3.csv")
    unique_clusters = df.new_cluster.value_counts()
    large_clusters = {i: c for i, c in unique_clusters.items() if c > 500}
    medium_clusters = {i: c for i, c in unique_clusters.items() if 100 <= c <= 500}
    small_clusters = {i: c for i, c in unique_clusters.items() if c <= 100}
    train_index_large, test_index_large = train_test_split(list(large_clusters.keys()), test_size=0.2, random_state=420)
    valid_index_large, test_index_large = train_test_split(test_index_large, test_size=0.5, random_state=420)
    train_index_medium, test_index_medium = train_test_split(list(medium_clusters.keys()), test_size=0.3,
                                                             random_state=420)
    valid_index_medium, test_index_medium = train_test_split(test_index_medium, test_size=0.5, random_state=420)
    train_index_small, test_index_small = train_test_split(list(small_clusters.keys()), test_size=0.4, random_state=420)
    valid_index_small, test_index_small = train_test_split(test_index_small, test_size=0.6, random_state=420)
    print(
        f"large_clusters  {len(large_clusters)}] - train {len(train_index_large)} (sum: {sum([v for k, v in large_clusters.items() if k in train_index_large])}) - valid {len(valid_index_large)} (sum: {sum([v for k, v in large_clusters.items() if k in valid_index_large])}) - test {len(test_index_large)} (sum: {sum([v for k, v in large_clusters.items() if k in test_index_large])})")
    print(
        f"medium_clusters  {len(medium_clusters)}] - train {len(train_index_medium)} (sum: {sum([v for k, v in medium_clusters.items() if k in train_index_medium])}) - valid {len(valid_index_medium)} (sum: {sum([v for k, v in medium_clusters.items() if k in valid_index_medium])}) - test {len(test_index_medium)} (sum: {sum([v for k, v in medium_clusters.items() if k in test_index_medium])})")
    print(
        f"small_clusters  {len(small_clusters)}] - train {len(train_index_small)} (sum: {sum([v for k, v in small_clusters.items() if k in train_index_small])}) - valid {len(valid_index_small)} (sum: {sum([v for k, v in small_clusters.items() if k in valid_index_small])})- test {len(test_index_small)} (sum: {sum([v for k, v in small_clusters.items() if k in test_index_small])})")
    df_train = df.loc[df["new_cluster"].isin(train_index_large + train_index_medium + train_index_small)]
    df_valid = df.loc[df["new_cluster"].isin(valid_index_large + valid_index_medium + valid_index_small)]
    df_test = df.loc[df["new_cluster"].isin(test_index_large + test_index_medium + test_index_small)]
    print(f"df_train (len: {len(df_train)}) - df_valid (len: {len(df_valid)}) - df_test (len: {len(df_test)})")
    df_train.to_csv("./data/stormy_data/train_v3.csv", index=False)
    df_valid.to_csv("./data/stormy_data/valid_v3.csv", index=False)
    df_test.to_csv("./data/stormy_data/test_v3.csv", index=False)


def split_crisisfacts_dataset():
    df = pd.read_csv("./data/crisisfacts_data/test_from_crisisfacts.csv")
    train_df = df.loc[df["event_type"].isin(["Hurricane Florence 2018", "Hurricane Sally 2020"])]
    valid_df = df.loc[df["event_type"].isin(["Hurricane Laura 2020", "Saddleridge Wildfire 2019"])]
    test_df = df.loc[df["event_type"].isin(
        ["2018 Maryland Flood", "Lilac Wildfire 2017", "Cranston Wildfire 2018", "Holy Wildfire 2018"])]
    train_df.to_csv("./data/crisisfacts_data/crisisfacts_train.csv", index=False)
    valid_df.to_csv("./data/crisisfacts_data/crisisfacts_valid.csv", index=False)
    test_df.to_csv("./data/crisisfacts_data/crisisfacts_test.csv", index=False)
    storm_df = df.loc[df["event_type"].isin(["Hurricane Florence 2018", "Hurricane Sally 2020", "Hurricane Laura 2020", "2018 Maryland Flood"])]
    storm_df.to_csv("./data/crisisfacts_data/crisisfacts_storm.csv", index=False)


def generate_diversified_random_pairs(df, multiplier, get_label, task):
    output_length = len(df) * multiplier
    sampled_df_list = []
    M = len(df)
    if M % 2 == 1:
      M -= 1
    num_labels = 3 if task == "event_temporality" else 2
    minimal_label_len = 0
    title = "title" if "title" in df.columns else "text"
    event = "wikidata_link" if "title" in df.columns else "event"
    time = "seendate" if "title" in df.columns else "unix_timestamp"
    url = "url" if "title" in df.columns else "source"

    while minimal_label_len < output_length/num_labels:
        B = random.sample(range(len(df)), M)
        A = list(zip(B[0:int(M/2)], B[int(M/2):M]))
        A_label = list(map(get_label, A))
        A_label_ignored_indics = [i for i, label in enumerate(A_label) if label == "ignored"]
        non_ignored_pairs = {i: a for i, a in enumerate(A) if i not in A_label_ignored_indics}
        sample_indics = list(non_ignored_pairs.keys())
        sentence_a = [df[title].values[A[pair][0]] for pair in sample_indics]
        sentence_b = [df[title].values[A[pair][1]] for pair in sample_indics]
        event_a = [df[event].values[A[pair][0]] for pair in sample_indics]
        event_b = [df[event].values[A[pair][1]] for pair in sample_indics]
        time_a = [df[time].values[A[pair][0]] for pair in sample_indics]
        time_b = [df[time].values[A[pair][1]] for pair in sample_indics]
        url_a = [df[url].values[A[pair][0]] for pair in sample_indics]
        url_b = [df[url].values[A[pair][1]] for pair in sample_indics]
        labels = [A_label[i] for i in sample_indics]
        sample_df = pd.DataFrame(
                list(zip(sentence_a, event_a, time_a, labels, sentence_b, event_b, time_b, url_a, url_b)),
                columns=['sentence_a', 'event_a', 'time_a', 'labels', 'sentence_b', 'event_b', 'time_b', 'url_a',
                         'url_b'])
        sampled_df_list.append(sample_df)
        final_df = pd.concat(sampled_df_list)
        final_df = final_df.drop_duplicates(keep='last')
        minimal_label_len = min(final_df.labels.value_counts().values)
        logger.info(f"minimal_label_len: {minimal_label_len}/{output_length/num_labels}")
    stratified_sample = final_df.groupby('labels').apply(
        lambda x: x.sample(n=floor(output_length/num_labels))
    )
    logger.info(f"stratified sampled df: {len(stratified_sample)}.")
    return stratified_sample


class StormyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 csv_path,
                 multiplier: int,
                 task: str = "combined",
                 data_type: str = "train",
                 forced: bool = True):
        random.seed(4)
        self.data_type = data_type
        self.task = task
        self.multiplier = multiplier
        self.df = pd.read_csv(csv_path)
        logger.info(f"Disc dataset: {task} - {data_type}.")
        logger.info(f"Unique sentence in original df: {len(self.df.title.unique())}.")
        self.label2int = self.get_label2int(task)
        if not Path(f"./data/stormy_data/{task}").exists():
            Path(f"./data/stormy_data/{task}").mkdir()
        save_path = str(Path(f"./data/stormy_data/{task}", f"{data_type}_v3.csv").absolute())
        self.sampled_df = self.get_sentence_pairs(save_path=save_path, forced=forced)
        logger.info(f"Unique sentence in sentence-pairs: {len(list(set(self.sampled_df.sentence_a.unique()).union(set(self.sampled_df.sentence_a.unique()))))}.")
        self.labels = [self.label2int[label] for label in self.sampled_df.labels.values]
        self.get_descriptions()

    @staticmethod
    def get_label2int(task):
        if task == "combined":
            label2int = {"different_event": 0, "earlier": 1, "same_date": 2, "later": 3}
        elif task == "event_deduplication":
            label2int = {"different_event": 0, "same_event": 1}
        elif task == "event_temporality":
            label2int = {"earlier": 0, "same_date": 1, "later": 2}
        else:
            ValueError(f"{task} not defined! Please choose from 'combined', 'event_deduplication' or 'event_temporality'")
        return label2int

    def get_sentence_pairs(self, save_path=None, forced=True):
        if not Path(save_path).exists() or forced:
            df = generate_diversified_random_pairs(self.df, self.multiplier, self.get_label, self.task)
            df.to_csv(save_path)
            logger.info(f"Sentence-pairs size: {len(df)}.")
            df["sentence_pairs"] = df["sentence_a"] + " " + df["sentence_b"]
            logger.info(f"# Unique Sentence-pairs size: {len(set(df['sentence_pairs']))}.")
            logger.info(f"samples labels distribution: {df.labels.value_counts()}).")
            logger.info(f"sentence pair df len: {len(df)})")
            logger.info(f"unique sentence_a len: {len(df.sentence_a.value_counts())})")
            logger.info(f"unique sentence_b len: {len(df.sentence_b.value_counts())})")
            logger.info(f"unique event_a len: {len(df.event_a.value_counts())})")
            logger.info(f"unique event_b len: {len(df.event_b.value_counts())})")
            logger.info(f"unique time_a len: {len(df.time_a.value_counts())})")
            logger.info(f"unique time_b len: {len(df.time_b.value_counts())})")
            logger.info(f"unique labels len: {len(df.labels.value_counts())})")
            logger.info("")
        else:
            df = pd.read_csv(save_path)

        logger.info(f"Sampled sentence-pairs' length: {len(df)}.")
        logger.info(f"Sampled sentence-pairs' label distribution: {df.labels.value_counts()}.\n")

        return df

    def get_label(self, index_tuple):
        clusters = self.df.wikidata_link.values
        cluster_i = clusters[index_tuple[0]]
        cluster_j = clusters[index_tuple[1]]
        if self.task == "combined":
            if cluster_i != cluster_j:
                return "different_event"
            else:
                unix_times = self.df.seendate.values
                unix_time_i = unix_times[index_tuple[0]][:8]
                unix_time_j = unix_times[index_tuple[1]][:8]
                if unix_time_i < unix_time_j:
                    label = "earlier"
                elif unix_time_i == unix_time_j:
                    label = "same_date"
                else:
                    label = "later"
                return label
        elif self.task == "event_deduplication":
            if cluster_i != cluster_j:
                return "different_event"
            else:
                return "same_event"
        elif self.task == "event_temporality":
            if cluster_i != cluster_j:
                return "ignored"
            else:
                unix_times = self.df.seendate.values
                unix_time_i = unix_times[index_tuple[0]][:8]
                unix_time_j = unix_times[index_tuple[1]][:8]
                if unix_time_i < unix_time_j:
                    label = "earlier"
                elif unix_time_i == unix_time_j:
                    label = "same_date"
                else:
                    label = "later"
                return label

    def get_descriptions(self):
        print(f"Disc dataset {({self.task}-{self.data_type})} csv has {len(self.sampled_df)} entries.")
        print(f"     Number of clusters - {len(self.df.wikidata_link.unique())}\n\n\n")

    def __getitem__(self, idx):
        return (self.sampled_df.sentence_a.values[idx], self.sampled_df.sentence_b.values[idx]), self.labels[idx]

    def __len__(self):
        return len(self.sampled_df)


class CrisisFactsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 csv_path,
                 multiplier: int,
                 task: str = "combined",
                 data_type: str = "train",
                 forced: bool = True):
        random.seed(4)
        self.data_type = data_type
        self.task = task
        self.multiplier = multiplier
        self.df = pd.read_csv(csv_path)
        logger.info(f"Crisisfacts dataset: {task} - {data_type}.")
        logger.info(f"Unique sentence in original df: {len(self.df.text.unique())}.")
        self.label2int = self.get_label2int(task)
        if not Path(f"./data/crisisfacts_data/{task}").exists():
            Path(f"./data/crisisfacts_data/{task}").mkdir()
        save_path = str(Path(f"./data/crisisfacts_data/{task}", f"{data_type}_v3.csv").absolute())
        self.sampled_df = self.get_sentence_pairs(save_path=save_path, forced=forced)
        logger.info(f"Unique sentence in sentence-pairs: {len(list(set(self.sampled_df.sentence_a.unique()).union(set(self.sampled_df.sentence_a.unique()))))}.")
        self.labels = [self.label2int[label] for label in self.sampled_df.labels.values]
        self.get_descriptions()

    @staticmethod
    def get_label2int(task):
        if task == "combined":
            label2int = {"different_event": 0, "earlier": 1, "same_date": 2, "later": 3}
        elif task == "event_deduplication":
            label2int = {"different_event": 0, "same_event": 1}
        elif task == "event_temporality":
            label2int = {"earlier": 0, "same_date": 1, "later": 2}
        else:
            ValueError(
                f"{task} not defined! Please choose from 'combined', 'event_deduplication' or 'event_temporality'")
        return label2int

    def get_sentence_pairs(self, save_path=None, forced=True):
        if not Path(save_path).exists() or forced:
            df = generate_diversified_random_pairs(self.df, self.multiplier, self.get_label, self.task)
            df.to_csv(save_path)
            logger.info(f"Sentence-pairs size: {len(df)}.")
            df["sentence_pairs"] = df["sentence_a"] + " " + df["sentence_b"]
            logger.info(f"# Unique Sentence-pairs size: {len(set(df['sentence_pairs']))}.")
            logger.info(f"samples labels distribution: {df.labels.value_counts()}).")
            logger.info(f"sentence pair df len: {len(df)})")
            logger.info(f"unique sentence_a len: {len(df.sentence_a.value_counts())})")
            logger.info(f"unique sentence_b len: {len(df.sentence_b.value_counts())})")
            logger.info(f"unique event_a len: {len(df.event_a.value_counts())})")
            logger.info(f"unique event_b len: {len(df.event_b.value_counts())})")
            logger.info(f"unique time_a len: {len(df.time_a.value_counts())})")
            logger.info(f"unique time_b len: {len(df.time_b.value_counts())})")
            logger.info(f"unique labels len: {len(df.labels.value_counts())})")
            logger.info("")
        else:
            df = pd.read_csv(save_path)

        logger.info(f"Sampled sentence-pairs' length: {len(df)}.")
        logger.info(f"Sampled sentence-pairs' label distribution: {df.labels.value_counts()}.\n")

        return df

    def get_label(self, index_tuple):
        clusters = self.df.event.values
        unix_times = self.df.unix_timestamp.values
        cluster_i = clusters[index_tuple[0]]
        cluster_j = clusters[index_tuple[1]]
        if self.task == "combined":
            if cluster_i != cluster_j:
                return "different_event"
            else:
                unix_time_i = self.round_to_day(unix_times[index_tuple[0]])
                unix_time_j = self.round_to_day(unix_times[index_tuple[1]])
                if unix_time_i < unix_time_j:
                    label = "earlier"
                elif unix_time_i == unix_time_j:
                    label = "same_date"
                else:
                    label = "later"
                return label
        elif self.task == "event_deduplication":
            if cluster_i != cluster_j:
                return "different_event"
            else:
                return "same_event"
        elif self.task == "event_temporality":
            if cluster_i != cluster_j:
                return "ignored"
            else:
                unix_time_i = self.round_to_day(unix_times[index_tuple[0]])
                unix_time_j = self.round_to_day(unix_times[index_tuple[1]])
                if unix_time_i < unix_time_j:
                    label = "earlier"
                elif unix_time_i == unix_time_j:
                    label = "same_date"
                else:
                    label = "later"
                return label

    @staticmethod
    def round_to_day(timestamp):
        return timestamp - (timestamp % 86400)

    def get_descriptions(self):
        print(f"Crisisfacts dataset{({self.task}-{self.data_type})} csv has {len(self.sampled_df)} entries.")
        print(f"     Number of clusters - {len(self.df.event.unique())}\n\n\n")

    def __getitem__(self, idx):
        return (self.sampled_df.sentence_a.values[idx], self.sampled_df.sentence_b.values[idx]), self.labels[idx]

    def __len__(self):
        return len(self.sampled_df)


if __name__ == "__main__":
    # split_stormy_dataset()
    # split_crisisfacts_dataset()

    train_csv_path = Path("./data/stormy_data/train_v3.csv")
    valid_csv_path = Path("./data/stormy_data/valid_v3.csv")
    test_csv_path = Path("./data/stormy_data/test_v3.csv")
    train_event_deduplication_storm = StormyDataset(train_csv_path, task="event_deduplication", multiplier=35, data_type="train", forced=True)
    valid_event_deduplication_storm = StormyDataset(valid_csv_path, task="event_deduplication", multiplier=30, data_type="valid", forced=True)
    test_event_deduplication_storm = StormyDataset(test_csv_path, task="event_deduplication", multiplier=30, data_type="test", forced=True)

    train_event_temporality_storm = StormyDataset(train_csv_path, task="event_temporality", multiplier=50, data_type="train", forced=True)
    valid_event_temporality_storm = StormyDataset(valid_csv_path, task="event_temporality", multiplier=30, data_type="valid", forced=True)
    test_event_temporality_storm = StormyDataset(test_csv_path, task="event_temporality", multiplier=30, data_type="test", forced=True)


    train_crisisfacts_csv_path = Path("./data/crisisfacts_data/crisisfacts_train.csv")
    valid_crisisfacts_csv_path = Path("./data/crisisfacts_data/crisisfacts_valid.csv")
    test_crisisfacts_csv_path = Path("./data/crisisfacts_data/crisisfacts_test.csv")
    train_event_deduplication_crisisfacts = CrisisFactsDataset(train_crisisfacts_csv_path, task="event_deduplication",
                                                               multiplier=23.3, data_type="train", forced=True)
    valid_event_deduplication_crisisfacts = CrisisFactsDataset(valid_crisisfacts_csv_path, task="event_deduplication",
                                                               multiplier=10, data_type="valid", forced=True)
    test_event_deduplication_crisisfacts = CrisisFactsDataset(test_crisisfacts_csv_path, task="event_deduplication",
                                                              multiplier=16, data_type="test", forced=True)

    train_event_temporality_crisisfacts = CrisisFactsDataset(train_crisisfacts_csv_path, task="event_temporality", multiplier=33, data_type="train", forced=True)
    valid_event_temporality_crisisfacts = CrisisFactsDataset(valid_crisisfacts_csv_path, task="event_temporality", multiplier=10, data_type="valid", forced=True)
    test_event_temporality_crisisfacts = CrisisFactsDataset(test_crisisfacts_csv_path, task="event_temporality", multiplier=16, data_type="test", forced=True)
