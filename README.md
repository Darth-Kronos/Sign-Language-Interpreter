[![Board Status](https://dev.azure.com/Mini-Project/a16ae0d8-ec9e-4a33-ba21-1fcbd57b18a5/411e1933-79f8-4d4e-8a17-7833098591d3/_apis/work/boardbadge/2a78519c-70cd-42ed-b8f6-6ed7da15ccbe)](https://dev.azure.com/Mini-Project/a16ae0d8-ec9e-4a33-ba21-1fcbd57b18a5/_boards/board/t/411e1933-79f8-4d4e-8a17-7833098591d3/Microsoft.RequirementCategory/)
# Sign-Language-Interpreter
## Overview
Sign language is an incredible advancement that brings the deaf and the dumb people close to our society. Unfortunately, some drawbacks have come along with this language. Not everyone knows how to interpret a sign language when having a conversation with a deaf and dumb person. One finds it hard to communicate without an interpreter. To solve this, we need an application that is easily available to all which converts the sign language to text in real-time. The main purpose of this project is to eliminate the barrier between the deaf and the dumb and the rest.
## Methodology
The signs are read using a webcam. The deaf and dumb person who is signing is made to stand in front of the webcam and the image captured from this is processed by skin segmentation and pose estimation which then decodes any sign language present by sending the pre-processed data to a neural network. The output of the neural network is again processed (Natural Language Processing) to convert the signs to text.
Our application will be built on Python and Tensorflow.

