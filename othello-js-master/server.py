from flask import Flask, request, json
from chainer import serializers
import sys, os
import pprint as pp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SLPolicy
import selector

# from montecarlo_policy import MontecarloPolicy

app = Flask(__name__, static_folder='.', static_url_path='')


# policy_path = '/home/mil/fukuta/work_space/aai/runs/0108_final/models/'
# # policy_path = './'
# policy = SLPolicy()
# # policy = MontecarloPolicy()
# serializers.load_hdf5(policy_path + 'sl_policy_10.model', policy)


@app.route('/')
def home():
    print("return index.html")
    return app.send_static_file('index.html')


@app.route('/initGame', methods=['POST'])
def init_game():
    return ""


@app.route('/getMove', methods=['POST'])
def face_info():
    if request.headers['Content-Type'] != 'application/json':
        print("json contents")
        print(request.headers['Content-Type'])
        return json.jsonify(res='error'), 400

    game_tree = request.get_json()
    # pp.pprint(game_tree)
    if game_tree['count'] == 1:
        print('init')
        selector.init(game_tree['aiType'])

    best_move = selector.select_move(game_tree)

    return json.jsonify(index=best_move)


'''
@app.route('/echo/<thing>')
def echo(thing):
    return thing
'''
app.run(host='0.0.0.0', port=8888, debug=True)
