import os
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, pitch, stream, interval, scale, duration
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from itertools import product

# Lista para guardar las notas extraídas
notes = []
notes_durations = []


def get_files(directory):
    global notes_durations
    notes_durations = []
    midi_files = [f for f in os.listdir(directory) if f.endswith('.mid')]

    for midi_file in midi_files:
        file_path = os.path.join(directory, midi_file)
        midi = converter.parse(file_path)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)

        if parts and parts.parts:
            notes_to_parse = parts.parts[0].recurse().notesAndRests
        else:
            notes_to_parse = midi.flat.notesAndRests

        for element in notes_to_parse:
            duration_value = element.duration.quarterLength
            if isinstance(element, note.Note):
                notes_durations.append(
                    (str(element.pitch), float(duration_value)))
            elif isinstance(element, chord.Chord):
                notes_durations.append(
                    ('.'.join(str(n) for n in element.normalOrder), float(duration_value)))
            elif element.isRest:
                notes_durations.append(('rest', float(duration_value)))

    # print('notes_durations', notes_durations)
    return notes_durations

# Función para cargar las notas desde un archivo "notes" previamente guardado


def load_notes(directory):
    try:
        with open("notes_durations", "rb") as filepath:
            notes_durations = pickle.load(filepath)
    except FileNotFoundError:
        notes_durations = get_files(directory)
    # print('notesDurationsLoaded', notes_durations)
    return notes_durations

# Obtener el número de notas diferentes en el archivo MIDI


def prepare_sequences(notes_durations, note_to_int, duration_to_int):
    sequence_length = 16
    network_input = []
    network_output = []

    for i in range(0, len(notes_durations) - sequence_length, 1):
        sequence_in = notes_durations[i:i + sequence_length]
        sequence_out = notes_durations[i + sequence_length]

        # Combinar la nota y la duración en un solo valor
        input_val = [note_to_int[note] + len(note_to_int) * duration_to_int[duration]
                     for note, duration in sequence_in]
        output_val = note_to_int[sequence_out[0]] + \
            len(note_to_int) * duration_to_int[sequence_out[1]]

        network_input.append(input_val)
        network_output.append(output_val)

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # Normalizar entrada y convertir salida a categórica
    normalized_input = network_input / \
        float(len(note_to_int) * len(duration_to_int))
    network_output = to_categorical(
        network_output, num_classes=len(note_to_int) * len(duration_to_int))

    return normalized_input, network_output


def train(network_input, network_output, note_to_int, duration_to_int, epochs):
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
    model.add(Dense(len(note_to_int) * len(duration_to_int), activation='softmax'))

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
    sequence_length = 8

    start = np.random.randint(0, len(network_input) - sequence_length)
    pattern = network_input[start]
    temperature = 1.8

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
    midi_stream = stream.Stream()

    for pattern, dur in prediction_output:
        # Crear nota o acorde basado en el patrón
        if '.' in pattern or pattern.isdigit():
            # Es un acorde
            notes_in_chord = pattern.split('.')
            chord_notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)
            new_chord = chord.Chord(
                chord_notes, duration=duration.Duration(dur))
            midi_stream.append(new_chord)
        else:
            # Es una nota
            new_note = note.Note(pattern, duration=duration.Duration(dur))
            new_note.storedInstrument = instrument.Piano()
            midi_stream.append(new_note)

    print("Contenido del stream:", midi_stream.show('text'))

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

    # Analizar la tonalidad del archivo MIDI
    key = midi_data.analyze('key')
    print("Tonalidad analizada:", key)

    # Crear la escala basada en la tonalidad y modo analizados
    if key.mode == 'major':
        scale_key = scale.MajorScale(key.tonic.name)
    else:
        scale_key = scale.MinorScale(key.tonic.name)

    # Obtener las notas de la escala
    scale_notes = [p.name for p in scale_key.getPitches()]

    # Función para encontrar la nota más cercana en la escala
    def closest_scale_pitch(note_pitch, scale_key):
        min_distance = float('inf')
        closest_pitch = None
        for scale_pitch in scale_key.getPitches('A0', 'C8'):
            distance = abs(scale_pitch.midi - note_pitch.midi)
            if distance < min_distance:
                min_distance = distance
                closest_pitch = scale_pitch
        return closest_pitch

    # Crear una nueva lista para elementos procesados
    processed_elements = []

    # Procesar todas las notas y acordes en todas las partes
    for element in midi_data.recurse():
        if isinstance(element, note.Note):
            # Ajustar notas
            if element.name not in scale_notes:
                closest_pitch = closest_scale_pitch(element.pitch, scale_key)
                element.pitch = closest_pitch
            processed_elements.append(element)
        elif isinstance(element, chord.Chord):
            # Ajustar acordes
            new_chord_pitches = [closest_scale_pitch(
                chord_note, scale_key) if chord_note.name not in scale_notes else chord_note for chord_note in element.pitches]
            new_chord = chord.Chord(new_chord_pitches)
            processed_elements.append(new_chord)

    # Crear un nuevo flujo de música y añadir los elementos procesados
    new_midi_data = midi_data.cloneEmpty()
    for element in processed_elements:
        new_midi_data.append(element)

    # Guardar el archivo MIDI ajustado
    new_midi_data.write('midi', fp=adjusted_midi_file_path)


if __name__ == '__main__':
    # Especifica el directorio que contiene los archivos MIDI
    midi_directory = "/home/southatoms/Desktop/developLinux/tensorFlow/src/assets/midiFiles"

    # Cargar las notas y duraciones del archivo "notes" (o crearlo a partir del archivo MIDI si no existe)
    notes_durations = load_notes(midi_directory)

    # Crear diccionarios para mapear notas y duraciones a enteros
    pitchnames = sorted(set(note for note, duration in notes_durations))
    durations = sorted(set(duration for note, duration in notes_durations))
    note_to_int = dict((note, number)
                       for number, note in enumerate(pitchnames))
    duration_to_int = dict((duration, number)
                           for number, duration in enumerate(durations))
    int_to_note = {i: (note, duration) for i, (note, duration)
                   in enumerate(product(pitchnames, durations))}

    # Obtener el número de notas diferentes en el archivo MIDI
    n_vocab = len(note_to_int) * len(duration_to_int)

    # Preprocesar las notas y duraciones y crear las secuencias de entrada y salida de la red neuronal
    network_input, network_output = prepare_sequences(
        notes_durations, note_to_int, duration_to_int)

    # Entrenar la red neuronal
    epochs = 5  # Número de épocas que quieres entrenar
    model = train(network_input, network_output,
                  note_to_int, duration_to_int, epochs)

    # Generar música utilizando el modelo
    prediction_output = generate_music(
        model, network_input, int_to_note, n_vocab)

    # Crear música y exportarla a un archivo MIDI
    create_music(prediction_output)
    # Asegúrate de reemplazar esto con la ruta correcta a tu archivo MIDI
    midi_file_path = 'output.mid'
    key = determine_key_from_midi(midi_file_path)
    adjusted_midi_file_path = 'adjusted_output.mid'  # Ruta al archivo MIDI ajustado
    print("La tonalidad del archivo MIDI es:", key)
    adjust_notes_to_key(midi_file_path, adjusted_midi_file_path)
    print("Archivo MIDI ajustado guardado en:", adjusted_midi_file_path)
 