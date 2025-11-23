import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import os


class NewsDataset(Dataset):
    """
        Class to load and preprocess the news dataset
    """
    def __init__(self,dataframe,tokenizer,max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.max_len = max_len
        self.tokenizer = tokenizer


    def __getitem__(self,index):
        title = str(self.data.iloc[index,0])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        return {
            'ids':torch.tensor(ids,dtype=torch.long),
            'mask':torch.tensor(mask,dtype=torch.long),
            'targets':torch.tensor(self.data.iloc[index,2],dtype=torch.long),
        }

    def __len__(self):
        return self.len


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

def calculate_correct(max_idx,targets):
    num_correct = (max_idx==targets).sum().item()
    return num_correct

def train(epoch,model,device,training_loader,optimizer,loss_function):
    tr_loss=0
    num_correct=0
    num_tr_steps=0
    num_tr_examples=0
    model.train()

    for i, data in enumerate(training_loader,0):
        ids = data["ids"].to(device,dtype=torch.long)
        mask = data["mask"].to(device,dtype=torch.long)
        targets = data["targets"].to(device,dtype=torch.long)

        outputs = model(ids,mask)

        loss = loss_function(outputs,targets)
        tr_loss += loss.item()
        max_val,max_idx = torch.max(outputs.data,dim=1)
        num_correct += calculate_correct(max_idx,targets)

        num_tr_steps += 1
        num_tr_examples += targets.size(0)

        if i%5000==0:
            loss_step = tr_loss/num_tr_steps
            acc_step = (num_correct*100)/num_tr_examples
            print(f"Loss per 5000 steps, steps {num_tr_steps} :{loss_step} ")
            print(f"Accuracy per 5000 steps, steps {num_tr_steps} :{acc_step} ")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / num_tr_steps
    epoch_acc = (num_correct*100)/num_tr_examples
    print(f"Training loss for epoch {epoch}:{epoch_loss}")
    print(f"Training Accuracy for epoch {epoch}:{epoch_acc}")

    return

def valid(epoch,model,device,testing_loader,loss_function):
    tr_loss=0
    num_correct=0
    num_tr_steps=0
    num_tr_examples=0
    model.eval()

    with torch.no_grad():
        for i,data in enumerate(testing_loader,0):
            ids = data["ids"].to(device,dtype=torch.long)
            mask = data["mask"].to(device,dtype=torch.long)
            targets = data["targets"].to(device,dtype=torch.long) 

            outputs = model(ids,mask)

            loss = loss_function(outputs,targets)
            tr_loss += loss.item()
            max_val,max_idx = torch.max(outputs.data,dim=1)
            num_correct += calculate_correct(max_idx,targets)

            num_tr_steps += 1
            num_tr_examples += targets.size(0)

            if i%1000==0:
                loss_step = tr_loss/num_tr_steps
                acc_step = (num_correct*100)/num_tr_examples
                print(f"Validation loss per 1000 steps, step {num_tr_steps}:{loss_step}")
                print(f"Validation accuracy per 1000 steps, step {num_tr_steps}:{acc_step}")

        epoch_loss = tr_loss / num_tr_steps
        epoch_acc = (num_correct*100)/num_tr_examples
        print(f"Validation loss for epoch {epoch}:{epoch_loss}")
        print(f"Validation Accuracy for epoch {epoch}:{epoch_acc}")

    return


def main():
    print("Starting...")
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_len", type=int, default=512)
    
    args = parser.parse_args()
    print("Reading and preprocessing data")
    s3_path = "s3://news-data-bucket-42/training/newsCorpora.csv"
    model_chkpt = "distilbert/distilbert-base-uncased"
    
    df = pd.read_csv(s3_path,sep="\t", names=["TITLE","URL","PUBLISHER","CATEGORY","STORY","HOSTNAME","TIMESTAMP"])
    df = df[["TITLE","CATEGORY"]]
    df = df.drop_duplicates()
    df["CATEGORY"] = df["CATEGORY"].map(
        {
            "b":"Business",
            "t":"Technology",
            "e":"Entertainment",
            "m":"Health"
        }
    )
    
    df = df.sample(frac=0.5,random_state=42)
    df = df.reset_index(drop=True)
    
    
    
    df["ENCODE_CAT"] = df["CATEGORY"].map({
        "Business":0,
        "Technology":1,
        "Entertainment":2,
        "Health":3
    })
    df = df.reset_index(drop=True)
    print(df.head())
    print("Loading Data and initializing model and tokenizer")
     
    train_size = 0.8
    train_dataset = df.sample(frac=train_size,random_state=42)
    test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset.reset_index(drop=True)
    
    print(f"full dataset:{df.shape}")
    print(f"train dataset: {train_dataset.shape}")
    print(f"test dataset:{test_dataset.shape}")
    
    EPOCHS = args.epochs
    TRAIN_BATCH_SIZE = args.train_batch_size
    VALID_BATCH_SIZE = args.valid_batch_size
    LEARNING_RATE = args.learning_rate
    MAX_LEN = args.max_len

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DistilBERTModel(model_chkpt=model_chkpt)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_chkpt)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
                                             
    training_set = NewsDataset(train_dataset,tokenizer,MAX_LEN)
    testing_set = NewsDataset(test_dataset,tokenizer,MAX_LEN)
    
    training_params = {
        'batch_size':TRAIN_BATCH_SIZE,
        'shuffle':True,
        'num_workers':0
    }
    testing_params = {
        'batch_size':VALID_BATCH_SIZE,
        'shuffle':True,
        'num_workers':0
    }
    
    training_loader = DataLoader(training_set, **training_params)
    testing_loader = DataLoader(testing_set, **testing_params)

    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        print(f"starting epoch:{epoch}")

        train(epoch, model,device, training_loader, optimizer, loss_function)
        valid(epoch, model, device, testing_loader, loss_function)

    output_dir = os.environ['SM_MODEL_DIR']
    output_model_file  = os.path.join(output_dir, "finetuned_distilbert_news.bin")
    
    torch.save(model.state_dict(), output_model_file)
    tokenizer.save_pretrained(output_dir)


if __name__=='__main__':
    main()
   