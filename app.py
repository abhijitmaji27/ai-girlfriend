from flask import Flask, request, jsonify
from chatbot import generate_character_response

app = Flask(__name__)

@app.route("/")
def read_root():
    return jsonify({"message": "Character API is LIVE!"})

@app.route("/character", methods=["POST"])
def character():
    try:
        question = request.json.get("question")
        answer = generate_character_response(question)
        return jsonify({"answer": answer})
    except Exception as e:
        # You can customize the error message and status code based on your application's needs
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
