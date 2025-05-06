from flask import Flask, render_template, request, redirect, url_for
import os
import google.generativeai as genai
from ml import photo_pred


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

API_KEY = 'AIzaSyCQmJPyQiNRa6wJQaodTkxt4eB_k9zxF34'
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    response = ""
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        user_file = request.files.get('file')

        if user_input:
            user_input = request.form['user_input']
            response = chat.send_message(user_input).text

        elif user_file:
            file = request.files['file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            x = photo_pred()  
            response = chat.send_message(f"{x} recipe").text
        else:
            response = "No input received"
    return render_template('chatbot.html', response=response)

if __name__ == '__main__':
    app.run(port=2000, debug=True)
