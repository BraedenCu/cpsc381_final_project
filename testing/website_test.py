from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>Test Site</title>
      </head>
      <body style="font-family: Arial; text-align: center; margin-top: 50px;">
        <h1>Flask Test Page</h1>
        <p>If you see this, your server is up and running!</p>
      </body>
    </html>
    """

if __name__ == '__main__':
    # Listen on all interfaces so you can connect from other machines
    app.run(host='0.0.0.0', port=5000, debug=True)