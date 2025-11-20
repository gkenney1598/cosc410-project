from openai import OpenAI
import pandas as pd
from pydantic import BaseModel
import csv
import ast

#build dataframe to get drug names to pass to model
drugs = pd.read_csv('data/drug_instructions.csv', encoding='utf-8')
drug_names = drugs[:]["name"]

#do not upload key!
client = OpenAI()

with open('data/example_drug.txt', 'r', encoding='utf-8') as file: #example drug with no important warning about usage
    example_drug_no_warning = file.read()

with open('data/example_drug_w_warning.txt', 'r', encoding='utf-8') as file: #example drug with important warning about usage
    example_drug_w_warning = file.read()

instructions = []

for i in range(0, 100, 10): #pass 10 drugs in at a time
    
    prompt_messages = [
        {"role": "system", "content": "You are a doctor writing instructions for drug usage. You should answer the following questions for each drug: "
            "Why is this medication prescribed? How should this medicine be used? What special precautions should I follow? What special dietary instructions should I follow? "
            "What should I do if I forget a dose? What side effects can this medication cause? Write this in this format if the drug does not need an"
            "important warning about usage: {example_drug}. Write in this format if it does: {example_drug_w_warning} Create a dictionary with each response"
            "with the structure {'name': <drug_name>, 'instructions': <instructions>}. Store all of the created dictionaries in a list"
            "Do not include bullet points or '-' in the instructions" },
        {"role": "user", "content": f"Write instructions for all of these drugs: {drug_names[i:i+10]}"}
    ]

    try:
        response = client.responses.parse(
            model="gpt-5",
            input=prompt_messages,
        )
        instruct = ast.literal_eval(response.output_text) #convert returned string to list of dicts
        instructions.append(instruct)

    except Exception as e:
        print("An error occurred:", e)


with open('data/generated_drug_instructions.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['name', 'instructions']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in instructions:
        writer.writerows(i for i in row)



