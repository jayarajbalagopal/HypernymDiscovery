# Supervised

### How to train the models
run `python3 train.py`   

On successful execution, three model.pt files will be generated in the current directory.

### Structure Explained
The models have been trained on 3 different tasks, 1A, 1B and 1C.

After training the models for appropriate tasks, copy the models into the corressponding task's directory.

Each of the task directory contains pickle files, please don't delete or overwrite them.

### Predict the Hypernyms
run `python3 predict.py`

On successful execution, the user is expected to provide an input hyponym and the model will ouput the top 10 matching hypernyms corressponding to the given input.

Certain part of the code needs to be uncommented in `predict.py` for loading the pretrained model, so uncomment any one of the lines (62,63,64) to load the model.

### Download pretrained models
In case you need to download the pretrained models insted of training them, download models from the following drive link
`https://drive.google.com/drive/folders/12yF-aNLq7tkO6l4jKqJ9-99LeOTeyl8K?usp=sharing`
and place them in the appropriate directories (1A, 1B or 1C).
