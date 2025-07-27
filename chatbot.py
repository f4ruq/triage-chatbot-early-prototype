import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from deep_translator import GoogleTranslator
from questions import hypertension, flu, heart_attack, stroke, pneumonia, arrhythmia, gastritis, diabeties, cancer, anaphylaxis

translator = GoogleTranslator(source="en", target="tr")

model_path = "./tunedbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

label_mapping = {
    0: "Hypertension", 1: "Flu", 2: "Heart Attack", 3: "Stroke",
    4: "Pneumonia", 5: "Arrhythmia", 6: "Gastritis",
    7: "Diabetes", 8: "Cancer", 9: "Anaphylaxis"
}

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1, top_k=None)

print("\nhastalık teşhisi için bir semptom metni giriniz.")
print("çıkmak için 'exit' yazın.\n")

while True:
    user_input = input("metin giriniz: ")
    if user_input.lower() == "exit":
        print("çıkılıyor...")
        break

    output = pipe(user_input)  

    best_prediction = output[0][0]  
    best_label_index = int(best_prediction['label'].replace("LABEL_", ""))
    best_label = label_mapping[best_label_index]
    best_score = best_prediction['score']

    print(best_label)

    print("\nTahminler:")
    for result in output[0]:  
        label_index = int(result['label'].replace("LABEL_", ""))
        predicted_label = label_mapping[label_index]
        score = result['score']
        print(f"{predicted_label}: {score:.2f}")
    
    #print(best_label)
    
    symptom_score = 0
    
    if(best_label=="Flu"):
        for question in flu:
            answer = input(question)
            if(answer == "yes"):
                symptom_score += 1
    
    elif(best_label=="Hypertension"):
        for question in hypertension:
            answer = input(question)
            if(answer == "yes"):
                symptom_score += 1
    
    elif(best_label=="Heart Attack"):
        for question in heart_attack:
            answer = input(question)
            if(answer.lower() == "yes"):
                symptom_score +=1
    
    elif(best_label=="Stroke"):
        for question in stroke:
            answer = input(question)
            if(answer.lower()=="yes"):
                symptom_score +=1
    
    elif(best_label=="Pneumonia"):
        for question in pneumonia:
            answer = input(question)
            if(answer=="yes"):
                symptom_score +=1
    
    elif(best_label=="Arrythmia"):
        for question in arrhythmia:
            answer = input(question)
            if(answer=="yes"):
                symptom_score +=1
    
    elif(best_label=="gastritis"):
        for question in gastritis:
            answer = input(question)
            if(answer=="yes"):
                symptom_score +=1
    
    elif(best_label=="diabeties"):
        for question in diabeties:
            answer = input(question)
            if(answer=="yes"):
                symptom_score +=1
    
    elif(best_label=="cancer"):
        for question in cancer:
            answer = input(question)
            if(answer=="yes"):
                symptom_score +=1

    elif(best_label=="anaphylaxis"):
        for question in anaphylaxis:
            answer = input(question)
            if(answer=="yes"):
                symptom_score +=1

    """if(symptom_score < 3):"""
        
    print("symptom score: ",symptom_score)

print("\n")
#print(label_mapping[0])
