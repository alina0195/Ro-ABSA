import re
import torch
from transformers import pipeline
from metric import compute_metrics
from helper import get_few_shot_prompt

def extract_aspects(input_string):
    aspects_list = ['delivery', 'environment', 'misc', 'price', 'product', 'promotions', 'quality', 'return warranty', 'security', 'service', 'shop diversity', 'shop organization', 'staff availability', 'staff competency', 'tech support']
    aspect_polarity_dict = {aspect: [] for aspect in aspects_list}
    pattern = r'aspects:([^=]+)'

    matches = re.findall(pattern, input_string)
    for match in matches:
        aspect_polarity_pairs = match.split(', ')
        for pair in aspect_polarity_pairs:
            clean_pair = re.sub(r'\s*\(.*?\)', '', pair).strip()
            try:
                aspect, polarity = clean_pair.split(':')
                aspect = aspect.strip()
                polarity = re.search(r'Positive|Negative|Neutral', polarity.strip(), re.IGNORECASE).group()
                if aspect in aspect_polarity_dict:
                    aspect_polarity_dict[aspect].append(polarity)
            except (ValueError, AttributeError):
                continue

    return aspect_polarity_dict
def zephyr_prompt(few_shot_df, subtask):
    prompt = ""
    if subtask == "ABSA":
        prompt = "Extract aspects and their associated polarities from provided reviews without altering the review text. Possible aspects: shop diversity, tech support, product, delivery, quality, staff competency, price, environment, return warranty, security, service, promotions, shop organization, staff availability, accessibility. "
    elif subtask == "ATC":
        prompt = "Extract aspects from provided reviews without altering the review text. Possible aspects: shop diversity, tech support, product, delivery, quality, staff competency, price, environment, return warranty, security, service, promotions, shop organization, staff availability, accessibility. "
    elif subtask == "ALSC":
        prompt = "The task is to extract the polarity for the provided aspect. I will provide you some examples, please do not change the input. Response should only contain the polarity which can be within these labels: positive, negative or neutral.  I will provide you some examples: "
    prompt += get_few_shot_prompt(few_shot_df, subtask)
    return prompt

def get_message(review, labeled_data, subtask):
    if subtask == "ABSA" or subtask == "ATC":
        return f"review: {review}"
    elif subtask == "ALSC":
        return f"review: {review} {labeled_data.split(':')[0]}"

def zephyr(evaluation_df, few_shot_df, subtask):
    pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16,
                    device_map="auto")

    true_labels_list = []
    pred_labels_list = []

    for index, row in evaluation_df.head(1).iterrows():
        messages = [
            {
                "role": "system",
                "content": zephyr_prompt(few_shot_df, subtask),
            },
            {"role": "user", "content": get_message(row['review'], row['labeled_data'], subtask)},
            {"role": "assistant", "content": get_message(row['review'], row['labeled_data'], subtask)}
        ]

        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = pipe(prompt, max_new_tokens=128, do_sample=False)
        true_labels, pred_labels = process_output(response, row['labeled_data'], subtask)
        true_labels_list.append(true_labels)
        pred_labels_list.append(pred_labels)
    compute_metrics(true_labels_list, pred_labels_list)

def process_output(response, labeled_data, subtask):
    outputs = str(response.choices[0].message)
    print(outputs)
    true_labels = ""
    pred_labels = ""

    review_split = str(outputs[0]["generated_text"]).split("<|assistant|>")[2]
    aspects = review_split.split("=========================")[0]

    if subtask == "ABSA":
        true_labels = labeled_data.split(', ')
        aspect_polarity_dict = extract_aspects(aspects)
        formatted_aspects = []
        for aspect, polarities in aspect_polarity_dict.items():
            if polarities:
                formatted_aspects.append(f"{aspect}:{', '.join(polarities)}")
        if not formatted_aspects:
            formatted_aspects = ["no aspects found"]
        pred_labels = ['; '.join(formatted_aspects)]

    elif subtask == "ATC":
        true_labels = labeled_data.split(', ')
        pred_labels = extract_aspects(aspects)
        if not pred_labels:
            pred_labels = ["no aspects found"]

    elif subtask == "ALSC":
        true_labels = labeled_data.split(': ')[1].strip()
        true_labels = true_labels.split(', ')
        pred_labels = ['positive' if 'positive' in aspects else 'negative' if 'negative' in aspects else 'neutral']

    return true_labels, pred_labels