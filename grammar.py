import torch
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Flask, render_template, request

# Load the model
tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")

grammar = Flask(__name__)

@grammar.route('/')
def home():
    return render_template('grammar.html')



#
def correct_grammar(sentence):
    inputs = tokenizer.encode(sentence, return_tensors="pt") # sentence라는 input이 들어오면 토크나이징 후 inputs에 바인딩
    outputs = model.generate(inputs, max_length=1024, early_stopping=True) # 디코딩
    corrected_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_sentence


@grammar.route('/correct_grammar', methods=['POST'])
def correct_grammar_api():
    sentence = request.form['sentence']
    corrected_sentence = correct_grammar(sentence)
    return render_template('grammar.html', sentence=sentence, corrected_sentence=corrected_sentence)

if __name__ == '__main__':
    grammar.run(debug=True)