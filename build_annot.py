from pathlib import Path
import json

PREFIX = "s3://terraform-workstations-bucket/jspaezp/20241022_prospect/"

with open("data/inventory.txt") as f:
    lines = f.readlines()

lines = [x.strip() for x in lines]

metadata_lines = [x for x in lines if x.endswith("_meta_data.parquet")]

partitions = {}
for x in metadata_lines:
    meta_file = x
    x = x.replace("_meta_data.parquet", "")
    matching = [
        PREFIX + y
        for y in lines
        if y.startswith(x)
        and y.endswith(".parquet")
        and not y.endswith("_meta_data.parquet")
    ]
    partitions[x] = {"metadata": PREFIX + meta_file, "files": matching}


with open("data/annot.json", "w") as f:
    json.dump(partitions, f, indent=4)

with open("data/partitions.txt", "w") as f:
    # f.writelines(list(partitions.keys()))
    for k in partitions.keys():
        f.write(k + "\n")
