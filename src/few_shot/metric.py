from sklearn.metrics import f1_score
import evaluate
def f1(pred, target):
    return f1_score(target, pred, average='weighted')

def compute_f1_score(true_labels, predicted_labels):
  new_formatted_labels_list_true = []
  new_formatted_labels_list_pred = []

  for label in true_labels:
      if len(label) > 1:
          formatted_string = "; ".join(label)
          formatted_string = formatted_string.split(", ")
          new_formatted_labels_list_true.append(formatted_string)
      else:
          new_formatted_labels_list_true.append(label)

  for label in predicted_labels:
      if len(label) > 1:
          formatted_string = "; ".join(label)
          formatted_string = formatted_string.split(", ")
          new_formatted_labels_list_pred.append(formatted_string)
      else:
          new_formatted_labels_list_pred.append(label)
  print(new_formatted_labels_list_true)
  print(new_formatted_labels_list_pred)

  last_true = []
  last_pred = []


  for index, value in enumerate(new_formatted_labels_list_true):
      last_true.append(new_formatted_labels_list_true[index][0])
      last_pred.append(new_formatted_labels_list_pred[index][0])

  f1_score_value = f1(last_true, last_pred)
  print(f"F1 score: {f1_score_value}")
  return f1_score_value

def recall(pred, target):
  pred = [p.strip() for p in pred]
  target = [p.strip() for p in target]

  sum = 0
  already_seen = []
  for p in pred:
    if p in target and p not in already_seen:
      sum += 1
      already_seen.append(p)
  sum=sum/(len(target))
  return sum

def compute_metrics(target, preds):
    other_metrics = evaluate.combine(["rouge", "meteor", "exact_match"])
    recalls_current = [recall(preds[idx],target[idx]) for idx in range(0,len(target))]
    preds_cleaned_combined = [' or '.join(e) for e in preds]
    targets_cleaned_combined = [' or '.join(t) for t in target]
    other_metrics_val = other_metrics.compute(predictions=preds_cleaned_combined, references=targets_cleaned_combined)
    print(f"F1-score: {compute_f1_score(target, preds)}")
    print(other_metrics_val)
    print(f"Recall: {sum(recalls_current)/len(recalls_current)}")