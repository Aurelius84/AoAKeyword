### AoAKWExtractor

Keywords Extractor by Attention over Attention with POS Mask Matrix.

### Requirements
+ pytorch
+ tensorboard
+ numpy
+ flask
+ gevent

### Train model
```
# install virtualenv
python3 -m venv ENV3.6
# activate env
source ENV3.6/bin/activate

python train.py

# tensorboard log
tensorboard --logdir=/runs --port 8089

# open your browser and visit
http://127.0.0.1:8089
```

### API
```
# start flask service with gevent in ENV
nohup python service.py &

# only support post with form data
# form data as followed:
{
    "title": "This is a title",
    "doc": "This is doc string"}

# response data
{
    "status": 200,
    "data":
    {
        "pre_kws":{
            "keyword1": prob1,
            "keyword2": prob2,
            "keyword3": prob3
        }
    }
}
```


