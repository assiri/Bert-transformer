# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ar-en")

text = """عقدت وزارة الصحة ورشة عمل لإعداد مدربين في مكافحة وضبط العدوى في المستشفيات، بدعم من صندوق الأمم المتحدة للسكان.
وقالت وزيرة الصحة د. مي الكيلة Mai Alkaila في كلمتها خلال الورشة إن مأسسة وحدة الجودة وسلامة المريض ركيزة أساسية في تحسين جودة الخدمات الصحية المقدمة في فلسطين، مضيفة أن وزارة الصحة تسعى مع كافة الشركاء في القطاع الصحي إلى رفع مستوى جودة الخدمات الصحية على كافة المستويات وفي كافة المرافق لتتمكن من أداء مهامها وبفعالية، استجابة لاحتياجات المواطنين بتوفير خدمات صحية ذات جودة عالية وآمنة.
"""

# Tokenize the text
batch = tokenizer.prepare_seq2seq_batch(src_texts=[text])

# Make sure that the tokenized text does not exceed the maximum
# allowed size of 512
batch["input_ids"] = batch["input_ids"][:, :512]
batch["attention_mask"] = batch["attention_mask"][:, :512]

# Perform the translation and decode the output
translation = model.generate(**batch)
tokenizer.batch_decode(translation, skip_special_tokens=True)

# ======================== using MBart (no Arabic fine-tuned models yet, but cc25 includes Arabic in training as one of the 25 languages)
# paper: https://arxiv.org/pdf/2001.08210.pdf
# docs: https://huggingface.co/transformers/master/model_doc/mbart.html
# fairseq: https://github.com/pytorch/fairseq/tree/master/examples/mbart

#!pip install -q git+https://github.com/huggingface/transformers.git

# ====================
# try En to Ar (proof of concept only, not fully trained, limited subset of OPUS Corpus - 100k pairs)

from transformers import AutoModelForSeq2SeqLM, MBartTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("akhooli/mbart-large-cc25-en-ar")
tokenizer = MBartTokenizer.from_pretrained("akhooli/mbart-large-cc25-en-ar")

article = "UN Chief Says There Is No Military Solution in Syria"
batch = tokenizer.prepare_seq2seq_batch(src_texts=[article], src_lang="en_XX")
translated_tokens = model.generate(
    **batch, decoder_start_token_id=tokenizer.lang_code_to_id["ar_AR"]
)
translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
translation

# Arabic to Eglish fine-tuned on a subset of OPUS Corpus - 100k pairs
from transformers import AutoModelForSeq2SeqLM, MBartTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("akhooli/mbart-large-cc25-ar-en")
tokenizer = MBartTokenizer.from_pretrained("akhooli/mbart-large-cc25-ar-en")

article = ["ويؤكد رئيس الأمم المتحدة أنه لا يوجد حل عسكري في سوريا"]
batch = tokenizer.prepare_seq2seq_batch(src_texts=article, src_lang="ar_AR")
translated_tokens = model.generate(
    **batch, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"]
)
translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

print(translation)
