import cv2
import os
from moviepy.editor import *
import gradio as gr
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Function to get POS tagging for better lemmatization
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Function for lemmatization
def get_lemma(word):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(word)
    tagged_words = pos_tag(tokens)
    lemmas = []
    for tagged_word in tagged_words:
        word = tagged_word[0]
        pos = tagged_word[1]
        wordnet_pos = get_wordnet_pos(pos)
        lemma = lemmatizer.lemmatize(word, pos=wordnet_pos)
        lemmas.append(lemma)
    return ' '.join(lemmas)

# Apply lemmatization to a full sentence
def apply_lemma_to_string(sentence):
    words = word_tokenize(sentence)
    lemmas = [get_lemma(word) for word in words]
    return ' '.join(lemmas)

# Function to parse words or letters
def parse_string(string, dataset):
    words = string.strip().split()
    parsed_list = []
    
    i = 0
    while i < len(words):
        # Try compound words first
        if i < len(words) - 1:
            compound = f"{words[i]} {words[i+1]}"
            if compound.lower() in dataset:
                parsed_list.append(compound)
                i += 2
                continue
        
        # If compound word not found, process single word
        current_word = words[i]
        if current_word.lower() in dataset:
            parsed_list.append(current_word)
        else:
            # If word not found, process character by character
            for char in current_word:
                parsed_list.append(char)
        i += 1
    
    return parsed_list

# Function to remove empty values from a list
def remove_empty_values(lst):
    return [x for x in lst if x]

# Flatten nested lists
def flatten_lists(lst):
    flat_list = []
    for i in lst:
        if isinstance(i, list):
            flat_list.extend(flatten_lists(i))
        else:
            flat_list.append(i)
    return flat_list

# Load dataset of videos (words and letters)
path = 'dataset'
videos = []
VideosNames = []
myList = os.listdir(path)

# Extract filenames
for cu_video in myList:
    VideosNames.append(os.path.splitext(cu_video)[0].replace("-", " ").lower())

print("Available words and letters:", VideosNames)

# Main function to convert text to sign language video
def texttoSign(text, output_path="combined.mp4"):
    text = text.lower() + " "
    text = apply_lemma_to_string(text)
    text = re.sub('[^a-z ]+', ' ', text)  # Keep space in character set
    
    # Parse words and letters
    listofwords = parse_string(text, VideosNames)
    listofwords = remove_empty_values(listofwords)
    
    print(f"Parsed words: {listofwords}")  # Debug print

    clips = []
    
    # Process each word or letter
    for item in listofwords:
        video_path = f"dataset/{item}.mp4"
        if os.path.exists(video_path):
            clip = VideoFileClip(video_path)
            clips.append(clip)

    if clips:
        result_clip = concatenate_videoclips(clips, method='compose')
        result_clip.write_videofile(output_path, fps=30)
        return output_path
    else:
        return None

# Only run Gradio interface if file is run directly
if __name__ == "__main__":
    demo = gr.Interface(
        fn=texttoSign,
        inputs="text",
        outputs="video",
        title="ASL Text To Sign",
        description="Convert text into American Sign Language (ASL) video",
        examples=[["hello world"], ["sign language"], ["good job"]]
    )
    demo.launch(debug=True)
