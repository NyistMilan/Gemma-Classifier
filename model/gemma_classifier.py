import os
from torch import nn
from torch.nn import Sigmoid
from transformers import AutoModelForSequenceClassification

class GemmaClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2, dropout=0.1, train=True):
        super().__init__()
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                        num_labels=num_labels, 
                                                                        token=os.getenv("HF_TOKEN"))

        if not train:
            for param in self.model.parameters():
                param.requires_grad = False

        self.sigmoid = Sigmoid()

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        predictions = (self.sigmoid(logits) > 0.5).float()
        return logits, predictions