from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return """
    <html>
    <head>
        <title>FocusIA - Home</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
                text-align: center;
                padding: 50px;
            }
            h1 {
                color: #333;
            }
            a {
                color: #007bff;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <h1>Hello, FocusIA!</h1>
        <p>Welcome to the homepage of FocusIA.</p>
        <p><a href="/about">Visit the About page</a></p>
    </body>
    </html>
    """

@app.route('/about')
def about():
    return """
    <html>
    <head>
        <title>FocusIA - About</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
                text-align: center;
                padding: 50px;
            }
            h1 {
                color: #333;
            }
            a {
                color: #007bff;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <h1>About FocusIA</h1>
        <p>This is the About page for FocusIA!</p>
        <p><a href="/">Back to Home</a></p>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run()
