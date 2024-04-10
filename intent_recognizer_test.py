import tensorflow as tf
import random
from keras.models import load_model

from flask import Flask

app = Flask(__name__)

# TODO: Fix this
model = load_model('myModel.keras')

def response(sentence):
    sent_tokens = []    
    # Split the input sentence into words
    words = sentence.split()
    # Convert words to their corresponding word indices
    for word in words:                                           
        if word in tokenizer.word_index:
            sent_tokens.append(tokenizer.word_index[word])
        else:
            # Handle unknown words
            sent_tokens.append(tokenizer.word_index['<unk>'])
    sent_tokens = tf.expand_dims(sent_tokens, 0)
    #predict numerical category
    pred = model(sent_tokens)    
    #category to intent
    pred_class = np.argmax(pred.numpy(), axis=1)                
    # random response to that intent
    return random.choice(
        response_for_intent[index_to_intent[pred_class[0]]]), index_to_intent[pred_class[0]]

@app.route('/')
def hello_world():
    print("Note: Enter 'quit' to break the loop.")   
    while True:                                                
        query = input('You: ')
        if query.lower() == 'quit':
            break
        bot_response, typ = response(query)
        return ('Bot: {} -- TYPE: {}'.format(bot_response, typ))
    
# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()