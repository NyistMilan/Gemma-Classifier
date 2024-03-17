from dataclasses import dataclass

@dataclass
class Scores:
    loss: float = None
    accuracy: float = None
    accuracy_by_labels: list = None
    precision: list = None
    recall: list = None
    f1: list = None