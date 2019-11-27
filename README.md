# Sign-Language-Interpreter
## Overview
Sign language is an incredible advancement that brings the deaf and the dumb people close to our society. Unfortunately, some drawbacks have come along with this language. Not everyone knows how to interpret a sign language when having a conversation with a deaf and dumb person. One finds it hard to communicate without an interpreter. To solve this, we need an application that is easily available to all which converts the sign language to text in real-time. The main purpose of this project is to eliminate the barrier between the deaf and the dumb and the rest.
## Methodology
The signs are read using a webcam. The deaf and dumb person who is signing is made to stand in front of the webcam and the image captured from this is processed by skin segmentation and pose estimation which then decodes any sign language present by sending the pre-processed data to a neural network. The output of the neural network is again processed (Natural Language Processing) to convert the signs to text.
Our application will be built on Python and Tensorflow.
