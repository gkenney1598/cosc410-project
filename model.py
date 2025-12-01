import pandas as pd
import random
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset

df = pd.read_csv("data/200_Drug_Instructions.csv")

df = df.rename(columns={"text": "sentence", "generated": "labels"})

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

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

train_ds = train_ds.remove_columns(["sentence", "__index_level_0__"])
test_ds = test_ds.remove_columns(["sentence", "__index_level_0__"])

#Set format for PyTorch
train_ds.set_format("torch")
test_ds.set_format("torch")

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
    eval_dataset=test_ds,
)

trainer.train()