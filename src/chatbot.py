import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import json

# MODEL and TOKENIZER
model_path = "./bert-medical/manual-stop"  
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# LABEL MAPPING
with open("./bert-medical/label_mapping.json", "r") as f:
    id2label = json.load(f)

# PIPELINE
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1, top_k=None)

print("\nHastalik teşhisi için bir semptom metni giriniz.")
print("Çıkmak için 'exit' yazın.\n")

while True:
    user_input = input("Metin giriniz: ")
    if user_input.lower() == "exit":
        print("cikiliyor...")
        break

    output = pipe(user_input)

    best_prediction = output[0][0]
    best_label_index = int(best_prediction["label"].replace("LABEL_", ""))
    best_label_name = id2label[str(best_label_index)]
    best_score = best_prediction["score"]

    print(f"\nTahmin edilen hastalık: {best_label_name} ({best_score:.2%})")

"""
    print("\nTahminler:")
    for result in output[0]:
        label_index = int(result["label"].replace("LABEL_", ""))
        disease_name = id2label[str(label_index)]
        score = result["score"]
        print(f"{disease_name}: {score:.2%}")
    print("\n")"""
