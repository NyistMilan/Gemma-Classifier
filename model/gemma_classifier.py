import os
from torch import nn
from torch.nn import Sigmoid
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

class GemmaClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2, train=True):
        super().__init__()

        peft_config = LoraConfig(
          r=64,
          lora_alpha=16,
          lora_dropout=0.10,
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
        base_model = prepare_model_for_int8_training(base_model)
        self.base_model = get_peft_model(base_model, peft_config)

        if not train:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.sigmoid = Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        logits = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits
        predictions = (self.sigmoid(logits) > 0.5).float()
        return logits, predictions