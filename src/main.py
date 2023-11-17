import os
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from music21 import stream

# Lista para guardar las notas extraídas
notes = []

def check_weights_exist(filepath):
    return os.path.exists(filepath)

def get_files(directory):
    global notes
    notes = []

    # Lista todos los archivos en el directorio
    midi_files = [f for f in os.listdir(directory) if f.endswith('.mid')]

    for midi_file in midi_files:
        file_path = os.path.join(directory, midi_file)

        midi = converter.parse(file_path)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)

        if parts:  # si hay partes de instrumentos
            notes_to_parse = parts.parts[0].recurse().notesAndRests
        else:
            notes_to_parse = midi.flat.notesAndRests

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes

def load_notes(directory):
    try:
        with open("notes", "rb") as filepath:
            notes = pickle.load(filepath)
    except FileNotFoundError:
        notes = get_files(directory)
    return notes

def prepare_sequences(notes):
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 16
    network_input = []
    network_output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length: i + sequence_length + 1]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out[0]])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(len(pitchnames))
    network_output = to_categorical(np.array(network_output))

    return network_input, network_output, note_to_int, pitchnames

def train(network_input, network_output, note_to_int, pitchnames, epochs=2):
    model = Sequential()
    model.add(LSTM(128, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    model.add(Dense(32))
    model.add(Dropout(0.3))
    model.add(Dense(network_output.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')

    model.fit(network_input, network_output, epochs=epochs, batch_size=350, callbacks=[checkpoint])

    model.load_weights(filepath)

    # Guardar el mapeo note_to_int e int_to_note
    with open('note_to_int.pickle', 'wb') as f:
        pickle.dump(note_to_int, f)
    int_to_note = {number: note for note, number in note_to_int.items()}
    with open('int_to_note.pickle', 'wb') as f:
        pickle.dump(int_to_note, f)

    return model

def generate_music(model, network_input, int_to_note, n_vocab):
    np.random.seed(42)
    sequence_length = 16

    start = np.random.randint(0, len(network_input) - sequence_length)
    pattern = network_input[start]
    temperature = 1

    prediction_output = []

    for note_index in range(200):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        prediction = np.log(prediction) / temperature
        exp_prediction = np.exp(prediction)
        prediction = exp_prediction / np.sum(exp_prediction)

        index = np.random.choice(range(n_vocab), size=1, p=prediction.flatten())[0]
        result = int_to_note[index]
        prediction_output.append(result)

        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_music(prediction_output):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if '.' in pattern:
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note()
                new_note.storedInstrument = instrument.Piano()
                try:
                    new_note.pitch.midi = int(current_note)
                    new_note.duration.quarterLength = np.random.choice([0.25, 0.5, 1.0])
                    notes.append(new_note)
                except ValueError:
                    pass
            chord_note = chord.Chord(notes)
            chord_note.offset = offset
            output_notes.append(chord_note)
        elif pattern.isdigit():
            new_note = note.Note()
            new_note.storedInstrument = instrument.Piano()
            try:
                new_note.pitch.midi = int(pattern)
                new_note.duration.quarterLength = np.random.choice([0.25, 0.5, 1.0])
                new_note.offset = offset
                output_notes.append(new_note)
            except ValueError:
                pass

        offset += np.random.choice([0.5, 1.0, 1.5, 2, 2.5, 3, 3.5, 4])

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output.mid')

    
if __name__ == '__main__':
    midi_directory = "C:/Users/southAtoms/Desktop/desarrollo/IAMidi/src/assets/midiFiles"
    weights_file = "weights.best.hdf5"

    notes = load_notes(midi_directory)
    network_input, network_output, note_to_int, pitchnames = prepare_sequences(notes)
    n_vocab = len(set(notes))

    if check_weights_exist(weights_file):
        model = Sequential()
        model.add(LSTM(128, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128))
        model.add(Dense(32))
        model.add(Dropout(0.3))
        model.add(Dense(len(pitchnames), activation='softmax'))

        model.load_weights(weights_file)

        # Cargar el mapeo note_to_int e int_to_note
        with open('note_to_int.pickle', 'rb') as f:
            note_to_int = pickle.load(f)
        with open('int_to_note.pickle', 'rb') as f:
            int_to_note = pickle.load(f)
    else:
        model = train(network_input, network_output, note_to_int, pitchnames)
        # Aquí también deberías cargar int_to_note después de entrenar
        with open('int_to_note.pickle', 'rb') as f:
            int_to_note = pickle.load(f)

    prediction_output = generate_music(model, network_input, int_to_note, n_vocab)
    create_music(prediction_output)
