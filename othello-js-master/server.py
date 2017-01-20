from flask import Flask, request, json
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import selector

app = Flask(__name__, static_folder='.', static_url_path='')


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
    time.sleep(0.1)
    return json.jsonify(index=best_move)


'''
@app.route('/echo/<thing>')
def echo(thing):
    return thing
'''
app.run(host='0.0.0.0', port=8888, debug=True)
