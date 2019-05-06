# Human-Emotion-Recognition
Detect human faces and classify the emotions among happy,sad,angry and many more. Human emotion recognition plays an important role in the relational relationship. The automatic recognition of emotions has been a functioning examination theme from early times. Hence, there are a few advances made in this field. Emotions are reflected from discourse, hand and signals of the body and through outward appearances. Henceforth extracting and understanding of emotion has a high significance of the collaboration among human and machine correspondence. This project depicts the advances made in this field and the different methodologies utilized for recognition of emotions. The main objective of the paper is to propose a real-time implementation of emotion recognition system.  

# Note: If you want to train the model again do the following (Skip this section if you want to run the predictor):
1. Download the dataset from the link: https://www.kaggle.com/deadskull7/fer2013
2. Go to Emotion Detection--->models
3. In models folder delete all the _mini_XCEPTION files and the _emotion_training.log file. Your models folder should now consist of         _pycache_ folder and cnn.py file.
4. Paste the downloaded dataset into Emotion Detection--->Dataset--->fer2013 folder.
5. Now run train_emotion_classifier.py file to start the training. This process may take time and it depends on you how much you want to train and till how many epochs.
6. After suffcient training a _mini_XCEPTION files will be generated in your Emotion Detection/models folder.
7. Open Single.py and Multiple.py and edit the emotion_model_path value in both files so that it contains the name of the latest _mini_XCEPTION file.
Latest  _mini_XCEPTION file would have the largest value among the other files. Ex: _mini_XCEPTION.102-0.66 has the value 102 for its epoch value.

# Run the Emotion Detector:
1. Run Single.py for single face detection or Multiple for multiple face detection.

View Prject Report more details.

# Note: You need to have some specific libraries installed on your system for Python else the program won't work. If you don't have the libraries installed, install them first before doing anything. The libraries name can be found in the code files with import syntax.
