import spacy
import re
nlp = spacy.load("en_core_web_sm")


def extract_tangible_objects(subtasks):
    
    # Set to store unique tangible objects
    tangible_objects = set()
    
    # List of words to exclude
    exclude = ['environment', 'place', 'room', 'surroundings', 'area', 'space', 'setting', 'site']
    
    # Analyze each subtask
    for task in subtasks:
        # Process the text
        doc = nlp(task)
        # Extract nouns
        for chunk in doc.noun_chunks:
            if all(token.text.lower() not in exclude for token in chunk):
                # Convert the chunk to lowercase string
                chunk_text = chunk.text.lower()
                # Remove "the " if it exists at the beginning of the string
                chunk_text = re.sub(r'^the ', '', chunk_text)
                tangible_objects.add(chunk_text)
    
    # Convert the set to a sorted list
    return list(set(tangible_objects))

# Example usage:
subtasks = [
    'transport the computer to the sofa',
    'explore the environment to locate the book',
    'explore the environment to locate the remote control',
    'transport the book to the sofa',
    'transport the remote control to the sofa'
]




out = extract_tangible_objects(subtasks)

print(out)