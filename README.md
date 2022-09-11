# N-gram language model.
This is a simple implementation of n-gram language model used for text generation. 
Model is trained on a corpora of [ArXiv](https://arxiv.org/) articles. 
## Usage info:
### train.py (estimated running time 2 min)
```
usage: usage: N-gram Language Model [-h] --input_dir INPUT_DIR --model MODEL

--input_dir -Location of the data directory containing arxivData.json
--model Location of the model file 
```
### generate.py (estimated running time 10 s)
```
usage: Generating text with pretrained N-gram Language Model [-h] --model MODEL [--prefix PREFIX] [--length LENGTH]

--model Location of the model file
--prefix  
--length  
```


### Example
#### Input
```
python3 train.py --input_dir "data/arxivData.json" --model "model.pkl"
python3 generate.py --model "model.pkl" --prefix "artificial neural" --length 20
```
#### 
Output
```
artificial neural network (cnn) has been carried out both their classification performance of inference queries can be seen as a simple lstm
```
