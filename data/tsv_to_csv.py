from email import header
import pandas as pd

task = "CoLA"
for name in ["train", "test", "dev"]:
    filename = "./glue_data/" + task + "/" + name
    dataset = pd.read_csv(filename + ".tsv", sep="\t", header=0)
    if name != "test":
        dataset.columns["a", "label", "c", "sentence"]
    else:
        dataset.columns["index", "sentence"]
    dataset2 = pd.DataFrame(dataset, columns=["sentence", "label"])
    dataset.to_csv(filename + ".csv", encoding="utf-8", index=False, sep=",")
