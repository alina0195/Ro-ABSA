import re
import torch
import transformers
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
def llama_prompt(few_shot_df, subtask):
    prompt = ""
    if subtask == "ABSA":
        prompt = """Extract the aspects and their polarities. Possible aspects: shop diversity, tech support, product, delivery, quality,
              staff competency, price, environment, return warranty, security, service, promotions, shop organization, staff availability, accessibility.
              Here are some examples: """
    elif subtask == "ATC":
        prompt = """Extract the aspects from users reviews. Possible aspects: shop diversity, tech support, product, delivery, quality,
              staff competency, price, environment, return warranty, security, service, promotions, shop organization, staff availability, accessibility.
              Here are some examples: """
    elif subtask == "ALSC":
        prompt = """Extract the polarity for the provided aspect. Polarity can be positive, negative or neutral.
      Here are some examples: """
    prompt += get_few_shot_prompt(few_shot_df, subtask)
    return prompt

def get_message(review, labeled_data, subtask):
    if subtask == "ABSA" or subtask == "ATC":
        return f"review: {review}"
    elif subtask == "ALSC":
        return f"review: {review} {labeled_data.split(':')[0].strip()}"

def llama(evaluation_df, few_shot_df, subtask):
    model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={
            "torch_dtype": torch.float16,
            "quantization_config": {"load_in_4bit": True},
            "low_cpu_mem_usage": True,
        },
    )

    true_labels_list = []
    pred_labels_list = []

    for index, row in evaluation_df.iterrows():
        messages = [
            {"role": "system", "content": llama_prompt(few_shot_df, subtask)},
            {"role": "user", "content": get_message(row['review'], row['labeled_data'], subtask)},
        ]

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=64,
            eos_token_id=terminators,
            do_sample=False,
            temperature=0.0
        )

        true_labels, pred_labels = process_output(outputs[0]['generated_text'], row['labeled_data'], subtask)
        true_labels_list.append(true_labels)
        pred_labels_list.append(pred_labels)
    compute_metrics(true_labels_list, pred_labels_list)

def process_output(response, labeled_data, subtask):
    aspects_list = [
        'delivery', 'environment', 'misc', 'price', 'product', 'promotions',
        'quality', 'return warranty', 'security', 'service', 'shop diversity',
        'shop organization', 'staff availability', 'staff competency', 'tech support'
    ]

    valid_polarities = ['Positive', 'Negative', 'Neutral']
    true_labels = []
    pred_labels = []
    if subtask == "ABSA":
        true_labels = labeled_data.split(', ')
        content = re.search(r"'role': 'assistant', 'content': '([^']*)'", response)
        if content:
            content = content.group(1)
            pattern = re.compile(r'(\b(?:' + '|'.join(aspects_list) + r')\b):\s*(Positive|Negative|Neutral)', re.IGNORECASE)
            for match in pattern.findall(content):
                keyword = match[0].strip().lower()
                polarity = match[1].strip()
                if keyword in [kw.lower() for kw in aspects_list] and polarity in valid_polarities:
                    pred_labels.append(f"{keyword}:{polarity}")
        if not pred_labels:
            pred_labels.append("No aspects found")

    elif subtask == "ATC":
        true_labels = labeled_data.split(', ')
        content = re.search(r"'role': 'assistant', 'content': '([^']*)'", response)
        if content:
            content = content.group(1)
            pattern = re.compile(r'\b(?:' + '|'.join(aspects_list) + r')\b', re.IGNORECASE)
            pred_labels = pattern.findall(content)
        if not pred_labels:
            pred_labels = ["No aspects found"]

    elif subtask == "ALSC":
        true_labels = labeled_data.split(':')[1].strip()
        true_labels = true_labels.split(', ')

        content = re.search(r"'role': 'assistant', 'content': '([^']*)'", response)
        if content:
            content = content.group(1)
            pattern = re.compile(r'\b(Positive|Negative|Neutral|positive|negative|neutral)\b', re.IGNORECASE)
            pred_labels = pattern.findall(content)

        if not pred_labels:
            pred_labels = ["No aspects found"]

    return true_labels, pred_labels