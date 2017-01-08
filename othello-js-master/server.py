from flask import Flask, request, json
from chainer import serializers
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SLPolicy
import selector

app = Flask(__name__,static_folder='.',static_url_path='')
policy = SLPolicy()
serializers.load_hdf5('../sl_policy_0.model', policy)

@app.route('/')
def home():
    print("return index.html")
    return app.send_static_file('index.html')

@app.route('/initGame', methods=['POST'])
def initGame():
    return ""


@app.route('/getMove', methods=['POST'])
def face_info():
    if request.headers['Content-Type'] != 'application/json':
        print("json contents")
        print(request.headers['Content-Type'])
        return json.jsonify(res='error'), 400

    gameTree = request.get_json()
    print(gameTree)
    print(hasattr(gameTree,"count"))
    if hasattr(gameTree,"count"):
        if getattr(gameTree,"count") == 1:
            selector.init()
    global policy
    bestMove = selector.selectMove(gameTree, policy)
    #bestMove = selector.selectMove(gameTree["board"],gameTree["moves"],gameTree["player"],gameTree["count"])


    return json.jsonify(index=bestMove)

'''
@app.route('/echo/<thing>')
def echo(thing):
    return thing
'''
app.run(port=8000,debug=True)
