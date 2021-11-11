""" This module generates notes for a midi file using the
    trained neural network """
import pickle
import time
import numpy
import tensorflow as tf
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Flatten
import keras.backend as K

def predict_to_2d(lista):
    lista = lista[0]
    len_pred = len(lista)
    new_list = []
    
    for i in range(0,len_pred):
        new_list.append([float(lista[i]),i])
        
    return new_list
    
def sort_array(lista):
        lista = numpy.array(sorted(lista,key=lambda l:l[0], reverse=True))
        return lista, lista[:, 0], lista[:, 1]
        
def argmax_array2d(array):
        result = max(array, key=lambda x: x[0])
        return result[0], result[1]
        
def beam_search(model, hip, step, initial_pattern, n_vocab, int_to_note):
        #print('start')
        pattern_list = []
        score_list = []
        outputs_list = []
        hip = hip
        step = step
        
        for h in range(hip):
            #print('h='+str(h))
            #print(len(initial_pattern))
            inside_pattern = initial_pattern
            if len(inside_pattern) > 100:
                inside_pattern = initial_pattern[1:len(initial_pattern)]
            #print(len(inside_pattern))
            hip_index = []
            h_score = 0
            
            for s in range(step):
                #print('s='+str(s))
                #print('main')
                #print(len(inside_pattern))
                prediction_input = numpy.reshape(inside_pattern, (1, len(inside_pattern), 1))
                prediction_input = prediction_input / float(n_vocab)

                prediction = model.predict(prediction_input, verbose=0)
                
                pred_array = predict_to_2d(prediction)
                
                sorted_array, probabilities, indices = sort_array(pred_array)
                
                if s == 0:
                    index = indices[h]
                    h_score += probabilities[h]
                    
                    result = int_to_note[index]
                    hip_index.append(result)
                    #print('s==0')
                    #print(len(inside_pattern))
                    inside_pattern.append(index)
                    #print(len(inside_pattern))
                    inside_pattern = inside_pattern[1:len(inside_pattern)]
                    #print(len(inside_pattern))
                    
                elif s == step-1:
                    prob, index = argmax_array2d(sorted_array)
                    
                    h_score += prob
                    h_score += h_score/step
                    result = int_to_note[index]
                    hip_index.append(result)
                    
                    #print('s == step')
                    #print(len(inside_pattern))
                    inside_pattern.append(index)
                    #print(len(inside_pattern))
                    inside_pattern = inside_pattern[1:len(inside_pattern)]
                    #print(len(inside_pattern))
                    
                    score_list.append(h_score)
                    pattern_list.append(inside_pattern)
                    outputs_list.append(hip_index)
                    
                else:
                    prob, index = argmax_array2d(sorted_array)
                    
                    h_score += prob
                    result = int_to_note[index]
                    hip_index.append(result)
                    #print('s')
                    #print(len(inside_pattern))
                    inside_pattern.append(index)
                    #print(len(inside_pattern))
                    inside_pattern = inside_pattern[1:len(inside_pattern)]
                    #print(len(inside_pattern))
        
        winner_index = score_list.index(max(score_list))

        return outputs_list[winner_index], pattern_list[winner_index]
        

def generate():
    """ Generate a piano midi file """
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab)
    start = time.time()
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    end = time.time()
    print(end - start)
    create_midi(prediction_output)

def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences = False))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Load the weights to each node
    model.load_weights('weights4.hdf5')

    return model

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []
    hip = 4 
    step = 3
    
    # generate 500 notes
    while (len(prediction_output) < 500):
        
        result, pattern_beam  = beam_search(model, hip, step, pattern, n_vocab, int_to_note)
        
        prediction_output.extend(result)

        pattern = pattern_beam

    return prediction_output

def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')

if __name__ == '__main__':
    generate()
