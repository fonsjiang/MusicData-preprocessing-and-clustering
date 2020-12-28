import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord, pitch
nt=[]
def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element,note.Note):
                notes.append(str(element.pitch)+' ')
            # if isinstance(element, pitch.Pitch):
            #     notes.append(str(element.name)+'')
            # elif isinstance(element,chord.Chord):
            #     notes.append('.'.join(str(n) for n in element.normalOrder))
            # elif isinstance(element,note.Note):
            #     notes.append(str(element.name)+'\n')
        # nt.append(notes)
        notess = "".join(notes)
    with open('data/notes.txt', 'w') as filepath:
         filepath.write(notess)
    filepath.close()

    return notess

if __name__ == '__main__':
    get_notes()
