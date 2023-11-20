import os
import pickle
import numpy as np
from music21 import converter, instrument, note, chord,pitch,stream,interval,scale
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint

# Lista para guardar las notas extraídas
notes = []


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

        # Verificar si hay partes de instrumentos y que no estén vacías
        if parts and parts.parts:
            notes_to_parse = parts.parts[0].recurse().notesAndRests
        else:
            notes_to_parse = midi.flat.notesAndRests

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    print('notes', notes)

    return notes


# Función para cargar las notas desde un archivo "notes" previamente guardado
def load_notes(directory):
    # Intentar abrir el archivo "notes"
    try:
        with open("notes", "rb") as filepath:
            notes = pickle.load(filepath)
    # Si el archivo no existe, crearlo a partir del archivo MIDI
    except FileNotFoundError:
        notes = get_files(directory)
    print('notesLoaded', notes)
    return notes


# Obtener el número de notas diferentes en el archivo MIDI
def prepare_sequences(notes):
    # Crear un diccionario que mapea notas a enteros
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number)
                       for number, note in enumerate(pitchnames))
    int_to_note = {number: note for note, number in note_to_int.items()}

    # Crear la secuencia de entrada y salida
    sequence_length = 16
    network_input = []
    network_output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length: i + sequence_length + 1]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out[0]])

    # Convertir las secuencias de entrada y salida en arrays numpy y normalizar los valores
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(len(pitchnames))
    network_output = to_categorical(np.array(network_output))

    return network_input, network_output, int_to_note


def train(network_input, network_output, epochs):
    # Definir la arquitectura de la red neuronal
    model = Sequential()
    model.add(LSTM(128, input_shape=(
        network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    model.add(Dense(32))
    model.add(Dropout(0.3))
    model.add(Dense(network_output.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Comprobar si ya existe un archivo "weights.best.hdf5"
    if os.path.exists("weights.best.hdf5"):
        # Cargar los pesos desde el archivo existente
        model.load_weights("weights.best.hdf5")
        print("Cargando pesos existentes desde 'weights.best.hdf5'")
    else:
        # Definir el checkpoint para guardar los pesos de la mejor época
        filepath = "weights.best.hdf5"
        checkpoint = ModelCheckpoint(
            filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')

        # Entrenar la red neuronal y guardar los pesos de la mejor época
        model.fit(network_input, network_output, epochs=epochs,
                  batch_size=350, callbacks=[checkpoint])

        print("Entrenamiento completo. Pesos guardados en 'weights.best.hdf5'")

    return model


def generate_music(model, network_input, int_to_note, n_vocab):
    # Utilizar una semilla aleatoria de notas
    np.random.seed(42)
    sequence_length = 16

    start = np.random.randint(0, len(network_input) - sequence_length)
    pattern = network_input[start]
    temperature = 1

    # Generar 500 notas
    prediction_output = []

    for note_index in range(200):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        # Predecir la siguiente nota utilizando el modelo
        prediction = model.predict(prediction_input, verbose=0)

        # Ajustar la distribución de probabilidad con la temperatura
        prediction = np.log(prediction) / temperature
        exp_prediction = np.exp(prediction)
        prediction = exp_prediction / np.sum(exp_prediction)

        # Muestrear la siguiente nota utilizando la distribución de probabilidad ajustada
        index = np.random.choice(
            range(n_vocab), size=1, p=prediction.flatten())[0]
        result = int_to_note[index]
        prediction_output.append(result)

        # Agregar la nueva nota a la semilla y descartar la nota más antigua
        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


def generate_music(model, network_input, int_to_note, n_vocab):
    # Utilizar una semilla aleatoria de notas
    np.random.seed(42)
    sequence_length = 16

    start = np.random.randint(0, len(network_input) - sequence_length)
    pattern = network_input[start]
    temperature = 1

    # Generar 500 notas
    prediction_output = []

    for note_index in range(200):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        # Predecir la siguiente nota utilizando el modelo
        prediction = model.predict(prediction_input, verbose=0)

        # Ajustar la distribución de probabilidad con la temperatura
        prediction = np.log(prediction) / temperature
        exp_prediction = np.exp(prediction)
        prediction = exp_prediction / np.sum(exp_prediction)

        # Muestrear la siguiente nota utilizando la distribución de probabilidad ajustada
        index = np.random.choice(
            range(n_vocab), size=1, p=prediction.flatten())[0]
        result = int_to_note[index]
        prediction_output.append(result)

        # Agregar la nueva nota a la semilla y descartar la nota más antigua
        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


def create_music(prediction_output):
    offset = 0  # Inicializar el offset para la primera nota
    output_notes = []

    # Iterar sobre cada elemento en la secuencia generada
    for i, pattern in enumerate(prediction_output):
        # Si el patrón es un acorde (indicado por la presencia de '.')
        if '.' in pattern:
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note()
                new_note.storedInstrument = instrument.Piano()
                try:
                    new_note.pitch.midi = int(current_note) + 60  # Transponer la nota 5 octavas arriba
                    notes.append(new_note)
                except ValueError:
                    pass
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        elif pattern.isdigit():  # Si el patrón es una nota individual
            new_note = note.Note()
            new_note.storedInstrument = instrument.Piano()
            try:
                new_note.pitch.midi = int(pattern) + 60  # Transponer la nota 5 octavas arriba
                new_note.offset = offset
                output_notes.append(new_note)
            except ValueError:
                pass

        # Calcular el offset para la siguiente nota
        if i < len(prediction_output) - 1:
            next_offset = offset + np.random.choice([0.5, 1.0, 1.5, 2, 2.5, 3, 3.5, 4])
            duration = next_offset - offset
        else:
            duration = 1.0  # Duración por defecto para la última nota

        # Asignar la duración a la nota o acorde actual
        if output_notes:
            output_notes[-1].duration.quarterLength = duration

        offset = next_offset if i < len(prediction_output) - 1 else offset + duration

    # Crear un objeto stream con todas las notas y acordes
    midi_stream = stream.Stream(output_notes)

    # Escribir el stream en un archivo MIDI
    midi_stream.write('midi', fp='output.mid')


def determine_key_from_midi(midi_file_path):
    # Cargar el archivo MIDI
    midi_data = converter.parse(midi_file_path)

    # Analizar la tonalidad utilizando music21
    key = midi_data.analyze('key')
    
    # Retornar la tonalidad como una cadena de texto
    return key.tonic.name + " " + key.mode

def adjust_notes_to_key(midi_file_path, adjusted_midi_file_path):
    # Cargar el archivo MIDI
    midi_data = converter.parse(midi_file_path)

    # Definir la escala de La menor natural
    la_minor_scale = scale.MinorScale('A')

    # Obtener las notas de la escala de La menor
    scale_notes = [p.name for p in la_minor_scale.getPitches('A2', 'A4')]

    # Función para encontrar la nota más cercana en la escala
    def closest_scale_pitch(note_pitch):
        min_distance = float('inf')
        closest_pitch = None
        for scale_pitch in la_minor_scale.getPitches('A0', 'C8'):
            distance = abs(scale_pitch.midi - note_pitch.midi)
            if distance < min_distance:
                min_distance = distance
                closest_pitch = scale_pitch
        return closest_pitch

    # Función para comparar dos acordes
    def are_chords_equal(chord1, chord2):
        if len(chord1.pitches) != len(chord2.pitches):
            return False
        return all(c1.midi == c2.midi for c1, c2 in zip(chord1.pitches, chord2.pitches))

    last_chord = None  # Guardar el último acorde procesado

    # Crear una nueva lista para elementos procesados
    processed_elements = []

    # Procesar todas las notas y acordes en todas las partes
    for element in midi_data.recurse():
        if isinstance(element, note.Note):
            if element.name not in scale_notes:
                closest_pitch = closest_scale_pitch(element.pitch)
                element.pitch = closest_pitch
            processed_elements.append(element)
        elif isinstance(element, chord.Chord):
            new_chord_pitches = [closest_scale_pitch(chord_note) if chord_note.name not in scale_notes else chord_note for chord_note in element.pitches]
            new_chord = chord.Chord(new_chord_pitches)
            
            # Verificar si el nuevo acorde es diferente del último acorde procesado
            if last_chord is None or not are_chords_equal(new_chord, last_chord):
                processed_elements.append(new_chord)
                last_chord = new_chord

    # Crear un nuevo flujo de música y añadir los elementos procesados
    new_midi_data = midi_data.cloneEmpty()
    for element in processed_elements:
        new_midi_data.append(element)

    # Guardar el archivo MIDI ajustado
    new_midi_data.write('midi', fp=adjusted_midi_file_path)




if __name__ == '__main__':
    # Especifica el directorio que contiene los archivos MIDI
    midi_directory = "/home/southatoms/Desktop/developLinux/tensorFlow/src/assets/midiFiles"

    # Cargar las notas del archivo "notes" (o crearlo a partir del archivo MIDI si no existe)
    #notes = load_notes(midi_directory)

    # Obtener el número de notas diferentes en el archivo MIDI
    #n_vocab = len(set(notes))

    # Preprocesar las notas y crear las secuencias de entrada y salida de la red neuronal
    #network_input, network_output, int_to_note = prepare_sequences(notes)

    # Entrenar la red neuronal
    #epochs = 2  # Número de épocas que quieres entrenar
    #model = train(network_input, network_output, epochs)

    # Generar música utilizando el modelo
    #prediction_output = generate_music(
     #   model, network_input, int_to_note, n_vocab)

    # Crear música y exportarla a un archivo MIDI
    #create_music(prediction_output)
    midi_file_path = 'output.mid'  # Asegúrate de reemplazar esto con la ruta correcta a tu archivo MIDI
    key = determine_key_from_midi(midi_file_path)
    adjusted_midi_file_path = 'adjusted_output.mid'  # Ruta al archivo MIDI ajustado
    print("La tonalidad del archivo MIDI es:", key)
    adjust_notes_to_key(midi_file_path, adjusted_midi_file_path)
    print("Archivo MIDI ajustado guardado en:", adjusted_midi_file_path)