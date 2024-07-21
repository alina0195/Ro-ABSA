from openai import OpenAI
from metric import compute_metrics
from helper import get_few_shot_prompt
import re

# client = OpenAI(api_key=API_KEY)

def gpt_prompt(few_shot_df, subtask):
    prompt = ""
    if subtask == "ABSA":
        prompt = """You are an excellent Romanian expert. The task is to extract the aspects and the associated polarities. I will provide you some examples, please do not change the text inside the "review" tag. Possible aspects: shop diversity, tech support, product, delivery, quality, staff competency, price, environment, return warranty, security, service, promotions, shop organization, staff availability, accessibility.  I will answer only with statements such as: """
    elif subtask == "ATC":
        prompt = """You are an excellent Romanian expert. The task is to extract the aspects from users reviews. I will provide you some examples, please do not change the text inside the "review" tag. Possible aspects: shop diversity, tech support, product, delivery, quality, staff competency, price, environment, return warranty, security, service, promotions, shop organization, staff availability, accessibility.  You will answer only with statements such as: """
    elif subtask == "ALSC":
        prompt = """You are an excellent Romanian expert. The task is to extract the polarity for the provided aspect. Polarity can be positive, negative or neutral. Here are some examples: """
    prompt += get_few_shot_prompt(few_shot_df, subtask)
    return prompt

def gpt(evaluation_df, few_shot_df, subtask):
    true_labels_list = []
    pred_labels_list = []

    for index, row in evaluation_df.iterrows():
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                     "content": gpt_prompt(few_shot_df, subtask)
                },
                {
                    "role": "user",
                    "content": f"review: {row['review']}"
                }
            ],
            temperature=0.0
        )
        true_labels, pred_labels = process_output(response, row['labeled_data'], subtask)
        true_labels_list.append(true_labels)
        pred_labels_list.append(pred_labels)
    compute_metrics(true_labels_list, pred_labels_list)

def process_output(response, labeled_data, subtask):
    outputs = str(response.choices[0].message)
    print(outputs)
    true_labels = ""
    pred_labels = ""

    if subtask == "ABSA":
        start_index = outputs.find("content")
        outputs = outputs[start_index:].split('\'')[1].split(':', 1)
        if len(outputs) > 1:
            outputs = outputs[1].strip()
        else:
            outputs = "no aspects found"
        true_labels = labeled_data.split(', ')
        pred_labels = outputs.split(', ')

    elif subtask == "ATC":
        pattern = r"content='([^']+)'"
        match = re.search(pattern, outputs)
        if match:
            aspects = match.group(1)
            if 'aspects:' in aspects:
                aspects = aspects.split(":")[1].strip()
            else:
                aspects = "no aspects found"
        else:
            aspects = "no aspects found"
        true_labels = labeled_data.split(', ')
        pred_labels = aspects.split(', ')

    elif subtask == "ALSC":
        outputs = str(response.choices[0].message)
        pattern = re.compile(r'\b(Positive|Negative|Neutral)\b', re.IGNORECASE)
        pred_labels = pattern.findall(outputs)
        true_labels = labeled_data.split(":")[1].strip()
        true_labels = true_labels.split(', ')

    return true_labels, pred_labels
