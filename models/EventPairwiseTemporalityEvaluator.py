import os
import csv
from pathlib import Path
from tqdm import tqdm

import numpy as np
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from torch.utils.data import DataLoader
from sentence_transformers.util import batch_to_device
import logging
import torch
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


class EventPairwiseTemporalityEvaluator(LabelAccuracyEvaluator):
    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model=None, write_csv: bool = True, write_predictions: bool = True):
        super().__init__(dataloader, name, softmax_model, write_csv)
        self.csv_file = "evaluation_"+name+"_results.csv"
        self.name = name
        self.write_predictions = write_predictions
        self.csv_headers = ["epoch", "steps", "accuracy", "macro_precision", "macro_recall", "macro_f1",
                                     "micro_precision", "micro_recall",  "micro_f1",
                                     "weighted_precision", "weighted_recall", "weighted_f1"]
        if self.softmax_model is not None:
            self.softmax_model = self.softmax_model.to(self.device)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        logger.info(f"Evaluator on device: {self.device}")
        total = 0
        correct = 0
        y_true = []
        y_predict = []

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluation on the "+self.name+" dataset"+out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate

        for step, batch in tqdm(enumerate(self.dataloader)):
            features, label_ids = batch
            y_true.append(label_ids)
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], self.device)
            label_ids = torch.tensor(label_ids, device=self.device)
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)
            y_predict.append(torch.argmax(prediction, dim=1).detach().cpu().numpy())
            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()
        y_true = np.concatenate(y_true)
        y_predict = np.concatenate(y_predict)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_predict, average='macro', zero_division=0)
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_true, y_predict, average='micro', zero_division=0)
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_predict, average='weighted', zero_division=0)
        accuracy = correct/total

        logger.info("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))
        logger.info(f"Macro metrics:")
        logger.info(f"    precision: {macro_precision}")
        logger.info(f"    recall: {macro_recall}")
        logger.info(f"    f1: {macro_f1}")
        logger.info(f"Micro metrics:")
        logger.info(f"    precision: {micro_precision}")
        logger.info(f"    recall: {micro_recall}")
        logger.info(f"    f1: {micro_f1}")
        logger.info(f"Weighted metrics:")
        logger.info(f"    precision: {weighted_precision}")
        logger.info(f"    recall: {weighted_recall}")
        logger.info(f"    f1: {weighted_f1}")

        if output_path is not None and self.write_csv:
            csv_path = Path(output_path, self.csv_file).absolute()
            if self.write_predictions:
                y_true.dump(Path(output_path, f"{self.name}_labels.pkl").absolute())
                y_predict.dump(Path(output_path, f"{self.name}_prediction.pkl").absolute())

            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy, macro_precision, macro_recall, macro_f1,
                                     micro_precision, micro_recall,  micro_f1,
                                     weighted_precision, weighted_recall, weighted_f1])
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy, macro_precision, macro_recall, macro_f1,
                                     micro_precision, micro_recall,  micro_f1,
                                     weighted_precision, weighted_recall, weighted_f1])

        return macro_f1

    @property
    def device(self):
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return "cpu"