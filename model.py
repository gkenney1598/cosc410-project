import pandas as pd
import random
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score

df = pd.read_csv("data/combined_drug_instructions.csv")

df = df.rename(columns={"instructions": "sentence", "generated": "labels"})

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=0)

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)
val_ds = Dataset.from_pandas(val_df)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

for index, row in df.iterrows():
    sentence = row["sentence"];
    beginning = random.randint(0, len(sentence) - 512)
    truncated_sentence = sentence[beginning:beginning + 512]
    df.at[index, "sentence"] = truncated_sentence

def tokenize(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds = train_ds.remove_columns(["sentence", "__index_level_0__"])
test_ds = test_ds.remove_columns(["sentence", "__index_level_0__"])
val_ds = val_ds.remove_columns(["sentence", "__index_level_0__"])

#Set format for PyTorch
train_ds.set_format("torch")
test_ds.set_format("torch")
val_ds.set_format("torch")

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=20,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

trainer.train()

train_preds, train_label_ids, train_metrics = trainer.predict(train_ds)
train_f1 = f1_score(train_df["labels"], train_label_ids)
print("TRAIN F1 SCORE: ", train_f1)
train_acc = accuracy_score(train_df["labels"], train_label_ids)
print("TRAIN ACCURACY SCORE: ", train_acc)
train_recall = recall_score(train_df["labels"], train_label_ids)
print("TRAIN RECALL SCORE: ", train_recall)

val_preds, val_label_ids, val_metrics = trainer.predict(val_ds)
val_f1 = f1_score(val_df["labels"], val_label_ids)
print("VAL F1 SCORE: ",  val_f1)
val_acc = accuracy_score(val_df["labels"], val_label_ids)
print("VAL ACCURACY SCORE: ", val_acc)
val_recall = recall_score(val_df["labels"], val_label_ids)
print("VAL RECALL SCORE: ", val_recall)

test_preds, test_label_ids, test_metrics = trainer.predict(test_ds)
test_f1 = f1_score(test_df["labels"], test_label_ids)
print("TEST F1 SCORE: ", test_f1)
test_acc = accuracy_score(test_df["labels"], test_label_ids)
print("TEST ACCURACY SCORE: ", test_acc)
test_recall = recall_score(test_df["labels"], test_label_ids)
print("TEST RECALL SCORE: ", test_recall)

