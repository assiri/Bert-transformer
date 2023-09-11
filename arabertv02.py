# Use a pipeline as a high-level helper
from transformers import pipeline
import torch

pipe = pipeline("fill-mask", model="aubmindlab/bert-base-arabertv02")
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
model = AutoModelForMaskedLM.from_pretrained("aubmindlab/bert-base-arabertv02")
inputs = tokenizer("عاصمة السعودية [MASK].", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(
    as_tuple=True
)[0]
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
print(tokenizer.decode(predicted_token_id))
