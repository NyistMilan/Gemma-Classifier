import os
from torch import nn
from torch.nn import Softmax
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class GemmaClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2, train=True, rank=64, alpha=16, lora_dropout=0.10):
        super().__init__()

        peft_config = LoraConfig(
          r=rank,
          lora_alpha=alpha,
          lora_dropout=lora_dropout,
          bias='none',
          inference_mode=False,
          target_modules=[
            'o_proj',
            'v_proj',
          ],
        )

        base_model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                        num_labels=num_labels,
                                                                        token=os.getenv("HF_TOKEN"))
        base_model = prepare_model_for_kbit_training(base_model)
        self.base_model = get_peft_model(base_model, peft_config)

        if not train:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.softmax = Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        logits = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits
        predictions = (self.softmax(logits) > 0.5).float()
        return logits, predictions