import torch
import os
import json
from transformers import AutoModel, AutoTokenizer

MAX_LEN = 512
model_chkpt = "distilbert/distilbert-base-uncased"

class DistilBERTModel(torch.nn.Module):
    """
        Class to define the classification model
    """
    def __init__(self,model_chkpt):
        super().__init__()

        self.layer1 = AutoModel.from_pretrained(model_chkpt)
        
        self.feed_forward = torch.nn.Linear(768,768)
        self.dropout = torch.nn.Dropout(0.1)
        
        self.classifier = torch.nn.Linear(768,4)

    def forward(self,input_ids, attention_mask):
        """
            function to feed forward tokenized inputs into the defined model layers
        """
        output_1 = self.layer1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:,0]
        pooler = self.feed_forward(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)

        return output


def model_fn(model_dir):
    print(f"Loading model from : {model_dir}")

    model = DistilBERTModel(model_chkpt)
    model_state_dict = torch.load(os.path.join(model_dir,"finetuned_distilbert_news.bin"), map_location=torch.device("cpu"))
    model.load_state_dict(model_state_dict)
    
    return model


def input_fn(request_body,request_content_type):
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        sentence = input_data['inputs']
        return sentence
    else:
        raise ValueError(f"Unsupported content type:{request_content_type}")


def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_chkpt)
    inputs = tokenizer(input_data, max_length = MAX_LEN, add_special_tokens=True, return_tensors="pt", truncation=True,padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs,dim=1)

    class_names = ["Business",
        "Technology",
        "Entertainment",
        "Health",]
    predicted_class = torch.argmax(probabilities,dim=1).item()
    probabilities = probabilities.cpu().numpy().tolist()
    predicted_label = class_names[predicted_class]

    return {"predicted_label":predicted_label, "probabilities":probabilities}


def output_fn(prediction,accept):
    if accept == "application/json":
        return json.dumps(prediction),accept
    else:
        raise ValueError(f"Unsupported accept type:{accept}")

        