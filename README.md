# QUAK
QUasi Anomalous Knowledge for Anomaly Detection and Tagging in High Energy Physics

<p align="leftr">
<img src="etc/quak-logo.png" height=200>
</p>

# Quasi Anomalous Knowledge: Searching for new physics with embedded knowledge

This repository is the official implementation of [Quasi Anomalous Knowledge: Searching for new physics with embedded knowledge](https://arxiv.org/abs/2011.03550). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train_script.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate QUAK performance on LHCO dataset, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

To evaluate QUAK performance on MNIST dataset, run:

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Citation

We are preparing a journal submission, in the meantime, please cite our paper from arxiv:

@article{Park:2020pak,
    author = "Park, Sang Eon and Rankin, Dylan and Udrescu, Silviu-Marian and Yunus, Mikaeel and Harris, Philip",
    title = "{Quasi Anomalous Knowledge: Searching for new physics with embedded knowledge}",
    eprint = "2011.03550",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "11",
    year = "2020"
}


## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
