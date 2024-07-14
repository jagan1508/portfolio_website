from flask import Flask, request, jsonify,render_template
from flask_scss import Scss
from flask_sqlalchemy import SQLAlchemy

from python_rag.rag import chatbot_response

#chatbot_response("tell me about him?")

app = Flask(__name__)
Scss(app)

app.config["SQLALCHEMY_DATABASE_URI"]="sqlite:///database.db"

db=SQLAlchemy(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def chat():
    user_message = request.form['message']
    bot_response = chatbot_response(user_message)
    return jsonify({'user_message': user_message, 'bot_response': bot_response})
if __name__ == '__main__':
    app.run(debug=True)