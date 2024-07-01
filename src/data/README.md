## ![Static Badge](https://img.shields.io/badge/Structure-HaluQuestQA-blue)

The dataset contains span-level annotations for errors in long-form answers as per our defined evaluation criteria:
1. **Question misconception**: False assumptions made within the given question. 
2. **Factuality**: Accuracy and correctness of the answer as per verifiable facts.
3. **Relevance**: Specificity and meaningfulness of the answer.
4. **Completeness**: Answer comprehensiveness ensuring all question aspects are addressed.
5. **References**: (Un)helpful examples, analogies, and external references (websites or links) in the answer.

A subset of the dataset is shown below, where given a question and two possible answers (human and GPT-4), the `{evaluation_criteria}_span` column indicates the error spans in the answer for the respective evaluation criteria and the error justifications are given in `{evaluation_criteria}_reason` column.

![HaluQuestQA](https://github.com/UKPLab/lfqa-hallucination/blob/master/images/haluquestqa_sample.png?raw=true)

---

## ![Static Badge](https://img.shields.io/badge/Structure-incomplete_ans_detection-blue)
This dataset consists of question-answer pairs with expert span-level annotations for ``completeness`` aspect, along with justifications. The dataset is used for training the incomplete answer detection model.

A subset of the dataset is shown below:

![incomplete_ans_data](https://github.com/UKPLab/lfqa-hallucination/blob/master/images/incomplete_ans_data.png?raw=true)

---

## ![Static Badge](https://img.shields.io/badge/Structure-preference_data-blue)
The preference dataset consists of a question with two possible answers: one from humans and the other from GPT-4. 
Expert annotators choose the better answer based on our defined evaluation criteria. The preferred responses are present in the `preferred_response` column and the rejected responses are present in the `rejected_response` column.  

This dataset is used for preference optimization.

A subset of the dataset is shown below:

![preference_data](https://github.com/UKPLab/lfqa-hallucination/blob/master/images/preference_data.png?raw=true)
