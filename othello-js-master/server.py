from flask import Flask, request, json
import selector
#from . import Selector

app = Flask(__name__,static_folder='.',static_url_path='')

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

    #print request.jso
    gameTree = request.get_json()
    #print(posted_data["test_value"])
    bestMove = selector.selectMove(gameTree["board"],gameTree["moves"],gameTree["player"])
    return json.jsonify(index=bestMove)

'''
@app.route('/echo/<thing>')
def echo(thing):
    return thing
'''
app.run(port=8000,debug=True)
