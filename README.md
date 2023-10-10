# IE_Toolkit

## Prepare
1. pip install -r requirements.txt
2. Download the model from XXX.

## Extract the entity and relation
1. prepare the data in a json file, which is a list with sentences. Below is an example.

[
  {
    "tokens": [ "An", "art", "exhibit", "at", "the", "Hakawati", "Theatre", "in", "Arab", "east", "Jerusalem", "was", "a", "series", "of", "portraits", "of", "Palestinians", "killed", "in", "the", "rebellion", "."
    ]
  },
]

2. change the dataset_path in configs/predict.conf to the path of your prepared data
3. python ./ie_tool.py predict --config configs/predict.conf
