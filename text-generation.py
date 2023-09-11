from transformers import pipeline

generator = pipeline("text-generation")
res = generator("In this course, we will teach you how to")
print(res)
