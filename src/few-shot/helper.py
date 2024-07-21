import json
import pandas as pd

def read_file(filename):
    with open(filename, "r", encoding="utf-8-sig") as file:
        return json.load(file)

def format_data(filename, subtask):
    annotated_data = read_file(filename)
    rows = []

    for annotated_object in annotated_data:
      aspects = annotated_object.get("category_aspect", [])
      aspects_list = [aspect.get("labels", [])[0] for aspect in aspects if "labels" in aspect]

      if subtask == "ABSA":
          polarities = annotated_object.get("polarity_aspect", [])
          sentiments_list = polarities if isinstance(polarities, list) else [polarities] * len(aspects_list)
          combined_data = [f"{aspect}:{sentiment}" for aspect, sentiment in zip(aspects_list, sentiments_list)]
          labeled_data = ', '.join(combined_data) if combined_data else "No aspects or polarities found"
          rows.append({'review': annotated_object["body"], 'labeled_data': labeled_data})

      elif subtask == "ATC":
          aspects_list = [aspect.get("labels", [])[0] for aspect in aspects if "labels" in aspect]
          labeled_data = ', '.join(aspects_list)
          rows.append({'review': annotated_object["body"], 'labeled_data': labeled_data})

      elif subtask == "ALSC":
          aspects = annotated_object.get("category_aspect", [])
          polarities = annotated_object.get("polarity_aspect", [])
          aspects_list = [aspect.get("labels", [])[0] for aspect in aspects if "labels" in aspect]
          sentiments_list = polarities if isinstance(polarities, list) else [polarities] * len(aspects_list)
          for idx, aspect in enumerate(aspects_list):
              rows.append({'review': annotated_object["body"], 'labeled_data': aspect + ': ' + sentiments_list[idx].lower()})

    df = pd.DataFrame(rows, columns=['review', 'labeled_data'])
    return df

def get_few_shot_prompt(df, subtask):
    prompt = ""
    for index, row in df.iterrows():
        review_text = row['review']
        labeled_data = row['labeled_data']
        if subtask == "ABSA" or  subtask == "ATC":
            prompt += f"review: {review_text} \naspects: {labeled_data}"
        elif subtask == "ALSC":
            prompt += f"review: {review_text} {labeled_data.split(': ')[0].strip()}: \n{labeled_data.split(':')[1].strip()}"

        if index != df.index[-1]:
            prompt += "\n=========================\n"
    return prompt

def process_list_of_lists(list_of_lists):
  processed_list = []
  for sublist in list_of_lists:
      new_sublist = []
      for item in sublist:
          item = item.replace(',', ';')
          parts = item.split(';')
          parts = [part.strip() for part in parts]
          parts.sort()
          new_sublist.append('; '.join(parts))
      processed_list.append(new_sublist)
  return processed_list

