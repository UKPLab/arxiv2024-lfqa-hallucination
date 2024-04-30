## Hallucinations in Long-form Question Answering 

This repository contains the code for 

* Creating an expert-annotated hallucination dataset for long-form question answering
* Methodologies to detect and mitigate hallucinations in long-form question answers


### Dataset
The dataset is available in the `src/data` folder. The dataset is available in 3 formats:

1. complete_data.csv: Contains the expert annotated data along with the hallucination detection labels, reasons and scores.
2. preference_data.csv: Contains the expert annotators' preferences for human and model answers. This dataset is used
for preference optimization.
3. incomplete_ans_detection_data.csv: Contains the expert annotated data for incomplete answers. This dataset is used for 
training the incomplete answer detection model. This is used as our error feedback model in the feedback-assisted refinement approach.
