import tensorflow as tf
import random
from keras.models import load_model

model = load_model('intentRecognizer.keras')

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

print("Note: Enter 'quit' to break the loop.")   
while True:                                                
    query = input('You: ')
    if query.lower() == 'quit':
        break
    bot_response, typ = response(query)
    print('Geek: {} -- TYPE: {}'.format(bot_response, typ))
    print()