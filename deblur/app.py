from flask import Flask, request , jsonify, render_template


app = Flask(__name__)
@app.route("/", methods=['GET'])


if __name__ == "__main__":
    # c1App = ClientApp()

    app.run(host='0.0.0.0', port=8080)