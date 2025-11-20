# Research Question

Can we train a model to predict whether or not a set of instructions for a drug is human-generated or AI-generated?

# Data

## Data Collection 

### Human-Generated Instructions

To begin, we created an excel spreadsheet and manually added 500 identification codes which we obtained for MedlinePlus. This is a government website that contains instructions for drug usage. We used the idenfitication codes to then scrape MedlinePlus and create a CSV with descriptions of all of these medications. This included sections such as "Why is this mediciation prescribed?", "How should this medication be used?", "Other uses for this medicine", "What special precautions should I follow?", and "What special dietary instruciton should I follow?"

### AI-Generated Instructions

Using the OpenAI API, we prompted ChatGPT-5 with the same 500 drugs and an example output of two medications we did not include in our dataset. We put all of the generated medication instructions into a CSV. 

## Data Organization
With our completed CSV files, we assigned 0s to all MedlinePlus drug instructions and 1s to all ChatGPT generated instructions.

# Model

We used the RoBERTa for Sequence Classification model. This model utilizes supervised learning with the preassigned labels to determine if it can accurately distinguish between Human-Generated instructions and AI-Generated instructions.

# Milestone 1

The results file output by the model from our first training run, with 100 AI generated and 100 human generated responses, is included in the repository as "results". Additionally, [here](https://drive.google.com/file/d/17HCmdBOQfpY1g8V73YuX72i6giQjD7F6/view?usp=sharing) is the terminal output from the training, including the loss. For this first pass, we were mainly concerned with providing a proof-of-concept and not how accurate our model was. To this end, we truncated the training texts to 256 characters. For Milestone II, we will work on fine-tuning the model to produce better results, including finding a work around to RoBERTa's token limit. However it is promising that the loss is generally around 0.1 without any fine-tuning, suggesting that the model can predict accurately a majority of the time.



