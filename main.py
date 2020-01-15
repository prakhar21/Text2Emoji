from flask import Flask, request, jsonify
from flask_cors import CORS
from flair.models import TextClassifier
from flair.data import Sentence
classifier = TextClassifier.load('./model/best-model.pt')

mapping = {
    'sad': '&#x1F61E',
    'smile': '&#x1F600',
    'food': '&#x1F37D',
    'heart': '&#10084;',
    'baseball': '&#x26be;'
}
app = Flask(__name__)
CORS(app)

@app.route('/emojify', methods=['POST'])
def emoji():
        data = request.form.get('text')
        if not len(data.strip()):
            return ''
        sentence = Sentence(data)
        classifier.predict(sentence)
        print (str(sentence.labels))
        if 'sad'in str(sentence.labels):
            return mapping['sad'] 
        elif 'smile' in str(sentence.labels):
            return mapping['smile'] 
        elif 'food' in str(sentence.labels):
            return mapping['food']
        elif 'heart' in str(sentence.labels):
            print (1)
            return mapping['heart']
        elif 'baseball' in str(sentence.labels):
            return mapping['baseball']

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
