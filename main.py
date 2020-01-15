from flask import Flask, request, jsonify
from flask_cors import CORS
from flair.models import TextClassifier
from flair.data import Sentence
classifier = TextClassifier.load('./model/best-model.pt')

app = Flask(__name__)
CORS(app)

@app.route('/emojify', methods=['POST'])
def emoji():
        data = request.form.get('text')
        sentence = Sentence(data)
        classifier.predict(sentence)
        return str(sentence.labels)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
