import csv
import os
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from torch import nn
from sentence_transformers.util import batch_to_device
from sklearn.metrics import precision_recall_fscore_support


logger = logging.getLogger(__name__)


class CustomSentenceTransformer(SentenceTransformer):

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        eval_path = output_path
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            eval_path = os.path.join(output_path, "eval")
            os.makedirs(eval_path, exist_ok=True)

        if evaluator is not None:
            score = evaluator(self, output_path=eval_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)
                    if not Path(output_path, "classifier").exists():
                        Path(output_path, "classifier").mkdir()
                    torch.save(evaluator.softmax_model.classifier.state_dict(), Path(output_path, "classifier"))


class InferenceModel(object):
    def __init__(self,
                 path_to_lm: str,
                 path_to_classifier: str,
                 name: str = "_",
                 write_predictions: bool = True):
        self.labels2int = self.get_label2int(path_to_lm)
        self.csv_file = "evaluation_" + name + "_results.csv"
        self.name = name
        self.write_predictions = write_predictions
        self.model = CustomSentenceTransformer(path_to_lm, device=self.device)
        self.classifier = nn.Linear(3 * 256, len(self.labels2int), device=self.device)
        self.classifier.load_state_dict(torch.load(path_to_classifier))
        self.model.eval()
        self.classifier.eval()

    def run(self, dataloader, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        self.model.eval()
        logger.info(f"Evaluator on device: {self.device}")
        total = 0
        correct = 0
        y_true = []
        y_predict = []

        dataloader.collate_fn = self.model.smart_batching_collate

        for step, batch in enumerate(dataloader):
            features, label_ids = batch
            y_true.append(label_ids)
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], self.device)
            label_ids = torch.tensor(label_ids, device=self.device)
            with torch.no_grad():
                _, prediction = self.classifier(features, labels=None)
            y_predict.append(torch.argmax(prediction, dim=1).detach().cpu().numpy())
            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()
        y_true = np.concatenate(y_true)
        y_predict = np.concatenate(y_predict)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_predict, average='macro',
                                                                                     zero_division=0)
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_true, y_predict, average='micro',
                                                                                     zero_division=0)
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_predict,
                                                                                              average='weighted',
                                                                                              zero_division=0)
        accuracy = correct / total

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

        if self.write_predictions:
            y_true.dump(Path(output_path, f"{self.name}_labels.pkl").absolute())
            y_predict.dump(Path(output_path, f"{self.name}_prediction.pkl").absolute())
        return macro_f1

    @staticmethod
    def get_label2int(path_to_classifier):
        if "combined" in path_to_classifier:
            label2int = {"different_event": 0, "earlier": 1, "same_date": 2, "later": 3}
        elif "event_deduplication" in path_to_classifier:
            label2int = {"different_event": 0, "same_event": 1}
        elif "event_temporality" in path_to_classifier:
            label2int = {"earlier": 0, "same_date": 1, "later": 2}
        else:
            ValueError(
                f"{path_to_classifier} not defined! Please choose from 'combined', 'event_deduplication' or 'event_temporality'")
        return label2int

    @property
    def device(self):
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return "cpu"