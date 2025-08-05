import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

dataset = load_dataset("csv", data_files="bert_ready.csv", split="train")

# %80 training, %20 test
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# label encoding
label_classes = sorted(set(dataset["label"])) 
label_mapping = {label: idx for idx, label in enumerate(label_classes)}

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_classes))

# tokenization
def tokenize_function(example):
    tokens = tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)
    tokens["labels"] = label_mapping[example["label"]]  
    return tokens

# tokenized dataset
tokenized_train = train_dataset.map(tokenize_function, batched=False, remove_columns=train_dataset.column_names)
tokenized_eval = eval_dataset.map(tokenize_function, batched=False, remove_columns=eval_dataset.column_names)

# model parameters
training_args = TrainingArguments(
    output_dir="./bert-medical",
    evaluation_strategy="epoch",    
    save_strategy="epoch",
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,  
    num_train_epochs=15, 
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    remove_unused_columns=False,
    report_to="none"
)

# training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval
)

try:
    trainer.train()
except KeyboardInterrupt:
    print("Eğitim manuel olarak durduruldu. Kaydediliyor...")
    trainer.save_model("./bert-medical/manual-stop")
    tokenizer.save_pretrained("./bert-medical/manual-stop")

model.save_pretrained("./tunedbert_tr")
tokenizer.save_pretrained("./tunedbert_tr")

print("Model eğitimi tamamlandı ve kaydedildi.")
