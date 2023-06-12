

from transformers import BertTokenizer, BertForMaskedLM
from flask import Flask, request, jsonify

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

app = Flask(__name__)

@app.route('/generate_title', methods=['POST'])
def generate_title():
    input_text = request.json['input_text']

    # Tokenize the input text
    input_tokens = tokenizer.encode(input_text, add_special_tokens=True, return_tensors='pt')

    # Generate the output tokens using the BERT model
    with torch.no_grad():
        output_tokens = model.generate(input_tokens)

    # Decode the output tokens to get the generated title
    generated_title = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # Return the generated title as a response
    return jsonify({'generated_title': generated_title})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 5000
CMD ["python", "app.py"]


docker run -d -p 5000:5000 research-paper-titles

{
    "input_text": "A comprehensive analysis of machine learning algorithms"
}

docker login
docker tag research-paper-titles username/research-paper-titles
docker push username/research-paper-titles














