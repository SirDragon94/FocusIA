from flask import Flask, request

app = Flask(__name__)

# HTML per la navbar comune
navbar = """
<div style="background-color: #007bff;">
    <a href="/" style="color: white; margin: 10px;">Home</a>
    <a href="/about" style="color: white; margin: 10px;">About</a>
    <a href="/contact" style="color: white; margin: 10px;">Contact</a>
</div>
"""

@app.route('/')
def hello():
    return f"""
    <html>
    <head>
        <title>FocusIA - Home</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
                text-align: center;
                padding: 50px;
                margin: 0;
            }}
            h1 {{
                color: #333;
            }}
        </style>
    </head>
    <body>
        {navbar}
        <h1>Hello, FocusIA!</h1>
        <p>Welcome to the homepage of FocusIA.</p>
    </body>
    </html>
    """

@app.route('/about')
def about():
    return f"""
    <html>
    <head>
        <title>FocusIA - About</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
                text-align: center;
                padding: 50px;
                margin: 0;
            }}
            h1 {{
                color: #333;
            }}
        </style>
    </head>
    <body>
        {navbar}
        <h1>About FocusIA</h1>
        <p>This is the About page for FocusIA!</p>
    </body>
    </html>
    """

@app.route('/contact')
def contact():
    return f"""
    <html>
    <head>
        <title>FocusIA - Contact</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
                text-align: center;
                padding: 50px;
                margin: 0;
            }}
            h1 {{
                color: #333;
            }}
        </style>
    </head>
    <body>
        {navbar}
        <h1>Contact FocusIA</h1>
        <p>This is the Contact page for FocusIA! Feel free to reach out.</p>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run()
