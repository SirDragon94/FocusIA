from flask import Flask, request

app = Flask(__name__)

# HTML per la navbar comune
navbar = """
<div style="background-color: #007bff; padding: 10px; text-align: center;">
    <a href="/" style="color: white; margin: 0 15px; text-decoration: none;">Home</a>
    <a href="/about" style="color: white; margin: 0 15px; text-decoration: none;">About</a>
    <a href="/contact" style="color: white; margin: 0 15px; text-decoration: none;">Contact</a>
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
            a {{
                color: #007bff;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
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
            a {{
                color: #007bff;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
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

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
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
                a {{
                    color: #007bff;
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            {navbar}
            <h1>Thank You, {name}!</h1>
            <p>We have received your message:</p>
            <p><strong>Email:</strong> {email}</p>
            <p><strong>Message:</strong> {message}</p>
            <p><a href="/">Back to Home</a> | <a href="/contact">Send another message</a></p>
        </body>
        </html>
        """
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
            a {{
                color: #007bff;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            form {{
                margin-top: 20px;
            }}
            label {{
                display: block;
                margin: 10px 0 5px;
            }}
            input, textarea {{
                width: 100%;
                max-width: 400px;
                padding: 8px;
                margin-bottom: 10px;
            }}
            button {{
                background-color: #007bff;
                color: white;
                padding: 10px 20px;
                border: none;
                cursor: pointer;
            }}
            button:hover {{
                background-color: #0056b3;
            }}
        </style>
    </head>
    <body>
        {navbar}
        <h1>Contact Us</h1>
        <p>Fill out the form below to get in touch!</p>
        <form action="/contact" method="POST">
            <label for="name">Name:</label>
            <input type="text" id="name" name="name" placeholder="Your name" required>
            <label for="email">Email:</label>
            <input type="email" id="email" name="email" placeholder="Your email" required>
            <label for="message">Message:</label>
            <textarea id="message" name="message" placeholder="Your message" rows="5" required></textarea>
            <button type="submit">Send Message</button>
        </form>
        <p><a href="/">Back to Home</a> | <a href="/about">About</a></p>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run()
