import json
import tarfile
import os

input_data = [
    {"inputs": "The new OnePlus 15 is the phone to beat in 2025"},
    {"inputs": "Tesla's latest earnings report signals a major shift in the EV market"},
    {"inputs": "Breakthrough weight-loss drug shows promising results in clinical trials"},
    {"inputs": "Marvel's next phase could redefine the superhero genre"},
    {"inputs": "Taylor Swift's record-breaking tour cements her as the decade's biggest artist"}
]

def create_json_files(data):
    for i,d in enumerate(data):
        filename  = os.path.join(os.path.dirname(__file__), "..", "data", f"input{i+1}.json")
        with open(filename,'w') as f:
            json.dump(d,f, indent=4)


def create_tar_file(input_files):
    filename = os.path.join(os.path.dirname(__file__), "..", "data","inputs.tar.gz")
    with tarfile.open(filename,"w:gz") as tar:
        for file in input_files:
            tar.add(file)


def main():
    create_json_files(input_data)
    input_files = [os.path.join(os.path.dirname(__file__), "..", "data", f"input{i+1}.json") for i in range(len(input_data))]
    create_tar_file(input_files)


if __name__=='__main__':
   main() 