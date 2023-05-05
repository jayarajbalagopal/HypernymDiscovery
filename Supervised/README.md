# SUPERVISED 

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