import tensorflow as tf
from transformers import AutoTokenizer, TFBertForPreTraining

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForPreTraining.from_pretrained("bert-base-uncased")
input_ids = tokenizer(
    "Hello, my dog is cute", add_special_tokens=True, return_tensors="tf"
)
# Batch size 1

outputs = model(input_ids)
prediction_logits, seq_relationship_logits = outputs[:2]
print(outputs)
