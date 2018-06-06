# utf-8
# @file service.py
# @Synopsis  
# @author Kevin, liujiezhangbupt@gmail.com
# @version 0.1
# @date 2018-06-05
import torch
import json
from collections import OrderedDict
import numpy as np
import jieba
from utils.Dict import Dict
from utils.AoAKW import AoAKW
from utils.process import Process, KWSample
from flask import Flask, request
from gevent.wsgi import WSGIServer

app = Flask(__name__)

word2idx = Dict(json.load(open('docs/word2idx.json', 'r')))
cate2idx = Dict(json.load(open('docs/cate2idx.json', 'r')))
process = Process(word2idx, cate2idx)

def initModel():

    state = torch.load("model/05-29-15:05:06_checkpoint.pth.tar")

    model = AoAKW(word2idx, dropout_rate=0.3, embed_dim=50, hidden_dim=50, n_class=92)
    model.load_state_dict(state["state_dict"])
    print("init model sucessfully.")
    return model

def dataWrapper(title:str, doc:str):
    data = {"title":[title],
            "doc":[doc],
            "kws":None,
            "topic":None}
    v_doc, v_title, _, _ = process.transform(data)

    return v_doc, v_title

def response(v_doc, v_title):

    topic_probs, kw_probs, atten_s = model(v_doc, v_title)
    topic_pred = np.argmax(topic_probs.data, axis=1)

    # top 3
    topk, indices = torch.topk(atten_s, 3, dim=1)
    indices_ = indices.data.view(-1, 3).numpy()
    v_doc_ = v_doc.data.numpy()[0]
    pre_kws =v_doc_[indices_][0]
    pre_kws = word2idx.convert2word(pre_kws)
    probs = topk.data.view(1, -1).numpy().tolist()[0][:len(pre_kws)]
    result = OrderedDict({
        "status": 200,
        "data":{
            # 'pre_topic': cate2idx.getWord(topic_pred[0]),
            'pre_kws': dict(zip(pre_kws, probs))
        }
    })
    return json.dumps(result)

# init model
model = initModel()

@app.route('/keywords', methods = ['POST'])
def extractKeywords():
    if request.method == 'POST':
        # try:
        title = request.form['title']
        doc = request.form['doc']
        
        # segment
        v_doc, v_title = dataWrapper(' '.join(jieba.lcut(title)), ' '.join(jieba.lcut(doc)))
        return response(v_doc, v_title)
        # except:
        #     return json.dumps({"status":504,
        #             "data":[],
        #             "msg":"title and doc should be formed."})
    else:
        return json.dumps({"status":404,
                "data":[],
                "msg":"Only support Get method."})


if __name__ == "__main__":
    # app.run(host="0.0.0.0",port=5000, threaded=True)
    http_server = WSGIServer(('', 5000),app)
    http_server.serve_forever()
