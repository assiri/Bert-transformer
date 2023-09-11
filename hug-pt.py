from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# X_train = [
#     "We are very happy to show you the © Transformers library.",
#     "We hope you don't hate it.",
# ]
# batch = tokenizer(
#     X_train, padding=True, truncation=True, max_length=512, return_tensors="pt"
# )
# with torch.no_grad():
#     outputs = model(**batch)
#     print(outputs)
#     predictions = F.softmax(outputs.logits, dim=1)
#     print(predictions)
#     labels = torch.argmax(predictions, dim=1)
#     print(labels)
#     labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
#     print(labels)
# save_directory = "saved"
# tokenizer.save_pretrained(save_directory)
# model.save_pretrained(save_directory)
# tokenizer = AutoTokenizer.from_pretrained(save_directory)
# model = AutoModelForSequenceClassification.from_pretrained(save_directory)


model_name = "oliverguhr/german-sentiment-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
X_train_german = [
    "nicht keinen auten Eraebnis",
    "Das war unfair",
    "Das ist gar nicht mal so gut",
    "Dicht so schlecht wie erwartet",
    "Das war gut!",
    "Sie fährt ein grünes Auto. ",
]
batch = tokenizer(
    X_train_german, padding=True, truncation=True, max_length=512, return_tensors="pt"
)
with torch.no_grad():
    outputs = model(**batch)
    label_ids = torch.argmax(outputs.logits, dim=1)
    print(label_ids)
    labels = [model.config.id2label[label_id] for label_id in label_ids.tolist()]
    print(labels)
