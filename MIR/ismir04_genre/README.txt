ISMIR04 Genre Identification task dataset


This is a collection of audio used for the Genre Identification task of the ISMIR 2004 audio description contest organized by the Music Technology Group (Universitat Pompeu Fabra). The audio for the task was collected from Magnatune, which contains a large amount of music licensed under Creative Commons licenses. The task of the contest was to classify a set of songs into genres, using the genre labels that Magnatune provided in their database.

Further information about the original contest and the contents of the dataset can be obtained from the following technical report:
Cano P, Gómez E, Gouyon F, Herrera P, Koppenberger M, Ong B, Serra X, Streich S, Wack N. ISMIR 2004 audio description contest. Barcelona: Universitat Pompeu Fabra, Music technology Group; 2006. 20 p. Report No.: MTG-TR-2006-02
http://hdl.handle.net/10230/34013

The original contest website can be found at http://ismir2004.ismir.net/genre_contest/

The dataset contains the audio tracks from following 8 genres: classical, electronic, jazz- & blues, metal-, punk, rock-, pop, world.
For the genre recognition contest, the data was grouped into 6 classes: classical, electronic, jazz-blues, metal-punk, rock-pop, world, where in some cases two genres were merged into a single class. Note that ground-truth files uses these 6 classes, however in some cases the data is organised by original genre.

AUDIO

The audio is in MP3 format. It is divided into three folders, representing different subsets of the collection. Each folder has 729 files, split into classes. The number of files in each category reflects the proportion of files in each category in Magnatune when the dataset was created. No track appears in more than one folder. 
- Training: files for generating a classification model, arranged by class.
- Development: A separate set of files for participants to test their model against. 
- Evaluation: originally a private subset, the files used to evaluate the accuracy of all submitted models

The training and development set each consist of:
classical: 320 files
electronic: 115 files
jazz_blues: 26 files
metal_punk: 45 files
rock_pop: 101 files
world: 122 files 

The evaluation set consists of 729 tracks with a similar distribution. 

METADATA

Each folder of audio has a corresponding folder containing metadata of the files in that folder. The metadata is included in a file, tracklist.csv which has the following headers:
class, artist, album, track, track number, file path
The evaluation tracklist file has an additional column representing the magnatune track id of the recording.
Due to the way that the data was collected and distributed for the challenge, the metadata for the development subset is anonymised.

LICENSING

The audio is licensed under a CC Attribution-NonCommercial-ShareAlike license (https://creativecommons.org/licenses/by-nc-sa/1.0/).


