from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, FocusIA!"

@app.route('/about')
def about():
    return "This is the About page for FocusIA!"

if __name__ == "__main__":
    app.run()
