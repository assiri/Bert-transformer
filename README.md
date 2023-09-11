Transformer model architecture
A transformer architecture consists of an encoder and decoder that work together. The attention mechanism lets transformers encode the meaning of words based on the estimated importance of other words or tokens.

[Transformer](https://towardsdatascience.com/transformers-141e32e69591)

<div dir="rtl">

بنية نموذج المحولات
تتكون بنية المحولات من جهاز تشفير ووحدة فك تشفير يعملان معًا. تتيح آلية الانتباه للمحولات تشفير معنى الكلمات بناءً على الأهمية المقدرة للكلمات أو الرموز المميزة الأخرى.
</div>
BERT 
Bidirectional Encoder Representations from Transformers:
 - Encoder Representations: language modeling system, pre-trained with unlabeled data. Then fine-tuning.
 - from Transformer: based on powerful LP algorithm. Defines the architecture of BERT.
 Bidirectional: uses with lef and right context when dealing with a word. Defines the training process.

BERT - intuition:
 - BERT's general idea
 - BERT's history: from RNNs to Transformer
 - BERT's architecture
 - BERT's pre-training
 Applications:
 - Use the tokenizer to process text data
 Use BERT as an embedding layer
 Finu-tune BERT, the core of your model
<div dir="rtl">
 
تمثيلات التشفير ثنائي الاتجاه من المحولات:
  - تمثيلات التشفير: نظام نمذجة اللغة، مُدرب مسبقًا ببيانات غير مُسمّاة. ثم الضبط الدقيق.
  - من المحول: يعتمد على خوارزمية LP القوية. يحدد بنية BERT.
  ثنائي الاتجاه: يستخدم مع السياق الأيسر والأيمن عند التعامل مع كلمة ما. يحدد عملية التدريب.

بيرت - الحدس:
  - فكرة بيرت العامة
  - تاريخ BERT: من RNNs إلى المحولات
  - عمارة بيرت
  - تدريب بيرت المسبق
  التطبيقات:
  - استخدم الرمز المميز لمعالجة البيانات النصية
  استخدم BERT كطبقة تضمين
  Finu-tune BERT، جوهر النموذج الخاص بك
  </div>

BERT, which stands for Bidirectional Encoder Representations from Transformers, is based on Transformers, a deep learning model in which every output element is connected to every input element, and the weightings between them are dynamically calculated based upon their connection.
Using this bidirectional capability, BERT is pre-trained on two different, but related, NLP tasks: Masked Language Modeling and Next Sentence Prediction.

The objective of Masked Language Model (MLM) training is to hide a word in a sentence and then have the program predict what word has been hidden (masked) based on the hidden word's context. The objective of Next Sentence Prediction training is to have the program predict whether two given sentences have a logical, sequential connection or whether their relationship is simply random.

[BERT](https://www.techtarget.com/searchenterpriseai/definition/BERT-language-model)

Image to Text
  Word embedding one-hot encoding
  Transfer learning
  
Old fashioned NLP
RNN seq2seq coder/decoder
Contex vector encoder/decoder . it says how the current state of decoder is related the goal input sequince. this improves the decoder phase.

All you need is Attention
Downside of RNNs: the sequential processing is not global enough, and loses information for long sentences.
 What if all we need is...ATTENTION

Attention in Transformer

Scaled-dot product
Main idea:
    2 sequences (equals in case of self attention), A and B
    let's see how each element from A is related to each element from B
    let's recombine A according to this
Before: a given sequence A (and a context B).
After: a new sequence where element i is a mix of elements from A that were related to element B.
Transformer model architecture
A transformer architecture consists of an encoder and decoder that work together. The attention mechanism lets transformers encode the meaning of words based on the estimated importance of other words or tokens. This enables transformers to process all words or tokens in parallel for faster performance, helping drive the growth of increasingly bigger LLMs.
hanks to the attention mechanism, the encoder block transforms each word or token into vectors further weighted by other words. For example, in the following two sentences, the meaning of it would be weighted differently owing to the change of the word filled to emptied:

He poured the pitcher into the cup and filled it.
He poured the pitcher into the cup and emptied it.
The attention mechanism would connect it to the cup being filled in the first sentence and to the pitcher being emptied in the second sentence.

The decoder essentially reverses the process in the target domain. The original use case was translating English to French, but the same mechanism could translate short English questions and instructions into longer answers. Conversely, it could translate a longer article into a more concise summary.

<div dir="rtl">
بنية نموذج المحولات
تتكون بنية المحولات من جهاز تشفير ووحدة فك تشفير يعملان معًا. تتيح آلية الانتباه للمحولات تشفير معنى الكلمات بناءً على الأهمية المقدرة للكلمات أو الرموز المميزة الأخرى. يتيح ذلك للمحولات معالجة جميع الكلمات أو الرموز المميزة بالتوازي للحصول على أداء أسرع، مما يساعد على دفع نمو دورات LLM الأكبر حجمًا بشكل متزايد.
بفضل آلية الانتباه، تقوم كتلة التشفير بتحويل كل كلمة أو رمز مميز إلى متجهات مرجحة بكلمات أخرى. على سبيل المثال، في الجملتين التاليتين، سيتم ترجيح المعنى بشكل مختلف بسبب تغيير الكلمة المملوءة إلى الفارغة:

وسكب الإبريق في الكأس وملأه.
فسكب الإبريق في الكأس وأفرغه.
ستقوم آلية الانتباه بربطه بالكوب الذي يتم ملؤه في الجملة الأولى وبالإبريق الذي يتم إفراغه في الجملة الثانية.

تقوم وحدة فك التشفير بشكل أساسي بعكس العملية في المجال الهدف. كانت حالة الاستخدام الأصلية هي الترجمة من الإنجليزية إلى الفرنسية، ولكن يمكن للآلية نفسها ترجمة الأسئلة والتعليمات الإنجليزية القصيرة إلى إجابات أطول. على العكس من ذلك، يمكن أن تترجم مقالة أطول إلى ملخص أكثر إيجازا.
</div>

Attention (Q,K,V)=softmax(QK^T/sqr(k^d))
QK^T example

BERT's architecture: stacks of encoders


BERT's inputs flexibility:
[CLS] + Sentence A + [SEP] + Sentence B  // classification token and separation between 2 ccentences

What shape for sentences?
  tokenization string -> tokens embeding-vectors

  He likes playing -> [He, likes,play##ing] [[0.173,000],....]


Bert's tokenizer: WordPiece tokenizer 30,522 "words"  
  tokenizer.toknize("I love Transformer")  
['I','love','transform', '##er']  

Pre-Training:  
Two phases:
  MLM (Masked Language Model) -> vector for -each token
  NSP (Next Sentence Prediction) -> a vector  for classification


Masked Language Model: How to apply bidirectional training with transformer?  
We mask words we want to predict  

BERT learns relations between words and importance of context  
Makes difference between homonyms  مرادفات

Next sentence prediction:  
Get a higher level understanding, from words to sentences  
Get access to more tasks, like question answering  

**Quick CNN expllanation**
  Convolution 
 (text)  input list of vectors * -> Feature Detector = -> Feature Map

```
FullTokenizer=bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_#-768_A-12/1",trainable False)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)
tokenizer.tokenize ("My dog loves strawberries.")
tokenizer.convert_tokens_to_ids (tokenizer.tokenize ("My dog loves strawberries."))
def encode_sentence(sent):
  return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))

data_inputs=[encode_sentence(sentence) for sentence in data_clean]
data_with_len=[[sent,data_labels[i],len(sentt)] for i,sent in enumerate(data_inputs)
     

max_seq_length = 128
train_batch_size = 32



# Get BERT layer and tokenizer:
# More details here: https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2
bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2', trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

tokenizer.wordpiece_tokenizer.tokenize('hi, how are you doing?')  
tokenizer.convert_tokens_to_ids(tokenizer.wordpiece_tokenizer.tokenize('hi, how are you doing?'))  

``` 
<div align="center">
    <img width="512px" src='https://drive.google.com/uc?id=1-SpKFELnEvBMBqO7h3iypo8q9uUUo96P' />
    <p style="text-align: center;color:gray">Figure 2: BERT Tokenizer</p>
</div>

# This provides a function to convert row to input features and label
``` 
def to_feature(text, label, label_list=label_list, max_seq_length=max_seq_length, tokenizer=tokenizer):
  example = classifier_data_lib.InputExample(guid=None,
                                             text_a = text.numpy(),
                                             text_b = None,
                                             label = label.numpy())
  feature = classifier_data_lib.convert_single_example(0, example, label_list, max_seq_length, tokenizer)
  return (feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id)

```
## Wrap a Python Function into a TensorFlow op for Eager Execution
```
def to_feature_map(text, label):
  input_ids, input_mask, segment_ids, label_id = tf.py_function(to_feature, inp=[text,label], Tout=[tf.int32, tf.int32, tf.int32, tf.int32])
  input_ids.set_shape([max_seq_length])
  input_mask.set_shape([max_seq_length])
  segment_ids.set_shape([max_seq_length])
  label_id.set_shape([])
  x = {'input_words_ids' : input_ids,
       'input_mask': input_mask,
       'input_type_id': segment_ids
       }  
  return (x, label_id)
  
```
## Create a TensorFlow Input Pipeline with tf.data
```
with tf.device('/cpu:0'):
  # train
  train_data = (train_data.map(to_feature_map,
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
  .shuffle(1000)
  .batch(32, drop_remainder=True)        
  .prefetch(tf.data.experimental.AUTOTUNE))

  # valid
  valid_data = (valid_data.map(to_feature_map,
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
  .batch(32, drop_remainder=True)        
  .prefetch(tf.data.experimental.AUTOTUNE))

```

## Add a Classification Head to the BERT Layer
<div align="center">
    <img width="512px" src='https://drive.google.com/uc?id=1fnJTeJs5HUpz7nix-F9E6EZdgUflqyEu' />
    <p style="text-align: center;color:gray">Figure 3: BERT Layer</p>
</div>  

## Building the model
```
def create_model():
  input_words_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_words_ids")
  input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
  input_type_id = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_type_id")
  pooled_output, sequence_output = bert_layer([input_words_ids, input_mask, input_type_id])
  drop = tf.keras.layers.Dropout(0.4)(pooled_output)
  output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(drop)
  model = tf.keras.Model(
           inputs = {
               'input_words_ids': input_words_ids,
               'input_mask': input_mask,
               'input_type_id': input_type_id          
           },
           outputs = output )
  return model
  ```
  ## Fine-Tune BERT for Text Classification
```
model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])
model.summary()  

tf.keras.utils.plot_model(model=model, show_shapes=True, dpi=76)


```   


##  Train model
``` 
epochs = 4  
history = model.fit(train_data,
                    validation_data=valid_data,
                    epochs=epochs,
                    verbose=1)  

```
## Evaluate the BERT Text Classification Model
``` 
import matplotlib.pyplot as plt

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()

  plot_graphs(history, 'loss')

  plot_graphs(history, 'binary_accuracy')

test_data = tf.data.Dataset.from_tensor_slices((sample_example, [1]*len(sample_example)))
test_data = test_data.map(to_feature_map).batch(1)
preds = model.predict(test_data)
threshold = 0.5
['positive' if pred >= threshold else 'negative' for pred in preds]

```  

[TFhub](https://tfhub.dev/callmemehdi/AraBERT/1)

``` 
import tensorflow as tf  
import tensorflow_hub as hub  

model = hub.load('https://tfhub.dev/callmemehdi/AraBERT/1')  
outputs = model(tf.convert_to_tensor([input]))  




```
## Arabic Natural Language Processing  
[Arabic Natural Language Processing](https://huggingface.co/spaces/aubmindlab/Arabic-NLP)


[huggingface](https://huggingface.co/spaces/aubmindlab/Arabic-NLP)

# Example

```
from transformers import pipeline

pipe = pipeline("fill-mask", model="aubmindlab/bert-base-arabertv02")

# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
model = AutoModelForMaskedLM.from_pretrained("aubmindlab/bert-base-arabertv02")
```   
# BertForMaskedLM 
[BertForMaskedLM ](https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertConfig)


[BERT Text Classification in a different language](https://www.philschmid.de/bert-text-classification-in-a-different-language)
