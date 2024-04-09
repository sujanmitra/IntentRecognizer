import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import keras
 
import warnings
warnings.filterwarnings('ignore')


with open('Intent Recognition/Intent.json', 'r') as f:
    data = json.load(f)
 
print(data.keys())
print(type(data['intents']))
print(len(data['intents']))
print(data['intents'][0].keys())
data['intents'][-1]

def clean(line):
    cleaned_line = ''
    for char in line:
        if char.isalpha():
            cleaned_line += char
        else:
            cleaned_line += ' '
    cleaned_line = ' '.join(cleaned_line.split())
    return cleaned_line


#list of intents
intents = []                                            
unique_intents = []
#all text data to create a corpus
text_input= []    
#dictionary mapping intent with appropriate response
response_for_intent = {}                                
for intent in data['intents']:
    #list of unique intents
    if intent['intent'] not in unique_intents:            
        unique_intents.append(intent['intent'])  
    for text in intent['text']:
        #cleaning is done before adding text to corpus
        text_input.append(clean(text))                    
        intents.append(intent['intent'])
    if intent['intent'] not in response_for_intent:
        response_for_intent[intent['intent']] = [] 
    for response in intent['responses']:
        response_for_intent[intent['intent']].append(response)


print("Intent :",intents[0])
print("Number of Intent:",len(intents))
print("Sample Input:", text_input[0])
print('Length of text_input:',len(text_input))
print("Sample Response: ", response_for_intent[intents[0]])


tokenizer = Tokenizer(filters='',oov_token='<unk>')
tokenizer.fit_on_texts(text_input)
sequences = tokenizer.texts_to_sequences(text_input)
padded_sequences = pad_sequences(sequences, padding='pre')
print('Shape of Input Sequence:',padded_sequences.shape)
padded_sequences[:5]


intent_to_index = {}
categorical_target = []
index = 0
 
for intent in intents:
    if intent not in intent_to_index:
        intent_to_index[intent] = index
        index += 1
    categorical_target.append(intent_to_index[intent])
 
num_classes = len(intent_to_index)
print('Number of Intents :',num_classes)
 
# Convert intent_to_index to index_to_intent
index_to_intent = {index: intent for intent, index in intent_to_index.items()}
index_to_intent


categorical_vec = tf.keras.utils.to_categorical(categorical_target, 
                                                num_classes=num_classes)
 
print('Shape of Ca',categorical_vec.shape)
categorical_vec[:5]

epochs=100
embed_dim=300
lstm_num=50
output_dim=categorical_vec.shape[1]
input_dim=len(unique_intents)
print("Input Dimension :{},\nOutput Dimension :{}".format(input_dim,output_dim))



model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, embed_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_num, dropout=0.1)),  
    tf.keras.layers.Dense(lstm_num, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(output_dim, activation='softmax')
])
 
optimizer = tf.keras.optimizers.Adam()  
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(padded_sequences, categorical_vec, epochs=epochs, verbose=0)



test_text_inputs = ["Hello", 
                    "my name is adam", 
                    "how are you?", 
                    "can you guess my name?",
                    "Do you get me","Adios"]
 
test_intents = ["Greeting",
                "GreetingResponse",
                "CourtesyGreeting",
                "CurrentHumanQuery",
                "UnderstandQuery",
                "GoodBye"]
 
test_sequences = tokenizer.texts_to_sequences(test_text_inputs)
test_padded_sequences = pad_sequences(test_sequences,  padding='pre')
test_labels = np.array([unique_intents.index(intent) for intent in test_intents])
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)
loss, accuracy = model.evaluate(test_padded_sequences, test_labels)

keras.models.save_model(model, "intentRecognizer.keras")