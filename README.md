# Research Question

Can we train a model to predict whether or not a set of instructions for a drug is human-generated or AI-generated?

## Data

### Data Collection 

#### Human-Generated Instructions

To begin, we created an excel spreadsheet and manually added 500 identification codes which we obtained for MedlinePlus. This is a government website that contains instructions for drug usage. We used the idenfitication codes to then scrape MedlinePlus and create a CSV with descriptions of all of these medications. This included sections such as "Why is this mediciation prescribed? How should this medication be used? Other uses for this medicine. What special precautions should I follow? What special dietary instruciton should I follow?

#### AI-Generated Instructions

Using the OpenAI API, we prompted ChatGPT-5 with the same 500 drugs and an example output of two medications we did not include in our dataset. We put all of the generated medication instructions into a CSV. 

### Data Organization
With our completed CSV files, we assigned 0s to all MedlinePlus drug instructions and 1s to all ChatGPT generated instructions.

## Model

We used _________ model. This model utilizes supervised learning with the preassigned labels to determine if it can accruately distinguish between Human-Generated instructions compared to AI-Generated instructions.



