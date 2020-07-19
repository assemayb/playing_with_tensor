import os
import re
import pickle
import yaml
import keras
import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras import layers, activations, models, preprocessing, utils


dir_path = "./data/questions"
files_list = os.listdir(dir_path + os.sep)

questions = []
answers = []

for filepath in files_list:
    stream = open(dir_path + os.sep + filepath, 'rb')
    docs = yaml.safe_load(stream)
    conversations = docs["conversations"]
    for con in conversations:
        if len(con) > 2:
            questions.append(con[0])
            replies = con[1:]
            ans = ""
            for rep in replies:
                ans += ' ' + rep
            answers.append(ans)
        elif len(con) > 1:
            questions.append(con[0])
            answers.append(con[1])

answers_with_tages = []
for i in range(len(answers)):
    if type(answers[i]) == str:
        answers_with_tages.append(answers[i])
    else:
        questions.pop(i)

answers = list()  # empty
for i in range(len(answers_with_tages)):
    answers.append("<START>" + answers_with_tages[i] + "<END>")


tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(questions + answers)
VOCAB_SIZE = len(tokenizer.word_index) + 1

# # PREPARING THE DATA FOR SEQ2SEQ MODEL
# vocab = []
# for word in tokenizer.word_index:
#     vocab.append(word)

# # encoder_input_data
tokenized_questions = tokenizer.texts_to_sequences(questions)
maxlen_questions = max([len(x) for x in tokenized_questions])
padded_questions = preprocessing.sequence.pad_sequences(
    tokenized_questions, maxlen=maxlen_questions, padding="post")
encoder_input_data = np.array(padded_questions)

# # decoder_input_data
tokenized_answers = tokenizer.texts_to_sequences(answers)
maxlen_answers = max([len(x) for x in tokenized_answers])
padded_answers = preprocessing.sequence.pad_sequences(
    tokenized_answers, maxlen=maxlen_answers, padding="post")
decoder_input_data = np.array(padded_answers)

# # decoder_output_data
tokenized_answers = tokenizer.texts_to_sequences(answers)
for i in range(len(tokenized_answers)):
    tokenized_answers[i] = tokenized_answers[i][1:]
padded_answers = preprocessing.sequence.pad_sequences(
    tokenized_answers, maxlen=maxlen_answers, padding='post')
onehot_answers = utils.to_categorical(padded_answers, VOCAB_SIZE)
decoder_output_data = np.array(onehot_answers)


# # #defining the encoder/decoder model

encoder_inputs = tf.keras.Input(shape=(maxlen_questions, ))
encoder_embedding = tf.keras.layers.Embedding(
    VOCAB_SIZE, 200, mask_zero=True)(encoder_inputs)
encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(
    200, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.Input(shape=(maxlen_answers,))
decoder_embedding = tf.keras.layers.Embedding(
    VOCAB_SIZE, 200, mask_zero=True)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(
    200, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(
    decoder_embedding, initial_state=encoder_states)

decoder_dense = tf.keras.layers.Dense(
    VOCAB_SIZE, activation=tf.keras.activations.softmax)
output = decoder_dense(decoder_outputs)

# model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
# model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss="categorical_crossentropy")
# # model.summary() #print the model summary

# # #Training the model
# model.fit([encoder_input_data, decoder_input_data], decoder_output_data, batch_size=50, epochs=10)
if keras.models.load_model("model.h5"):
    model = keras.models.load_model
else:
    model.save("model.h5")

# model = keras.models.load_model("model.h5")


def make_interference_models():
    # first the encoder model
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    decoder_state_input_h = tf.keras.Input(shape=(200, ))
    decoder_state_input_c = tf.keras.Input(shape=(200, ))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs
    )

    # the decoder model
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )
    return encoder_model, decoder_model

# #chatbot


def str_to_tokens(sentence):
    words = sentence.lower().split()
    tokens_list = []
    for word in words:
        tokens_list.append(tokenizer.word_index[word])
    return preprocessing.sequence.pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post')


enc_model, dec_model = make_interference_models()
for _ in range(10):
    states_values = enc_model.predict(
        str_to_tokens(input("Enter a question: ")))
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index["start"]
    stop_condition = False
    decoded_translation = ""
    while not stop_condition:
        dec_outputs, h, c = dec_model.predict(
            [empty_target_seq] + states_values)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                decoded_translation += ' {}'.format(word)
                sampled_word = word
        if sampled_word == "end" or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]

    print(decoded_translation)
