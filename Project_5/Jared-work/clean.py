import re

# Load the text file
with open('C:/Users/jared/Documents/GitHub/CSE450-Team/Project_5/Jared-work/austen.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Remove chapter headings (case-insensitive)
text = re.sub(r'(?i)chapter\s+[IVXLCDM0-9]+', '', text)

# Remove volume headings (case-insensitive)
text = re.sub(r'(?i)volume\s+[IVXLCDM0-9]+', '', text)

# Remove words with underscores
text = re.sub(r'_\w+_', '', text)

# Remove non-Unicode characters
text = re.sub(r'[^\x00-\x7F]+', '', text)

# Ensure that any two words are separated by exactly one space
text = re.sub(r'\s+', ' ', text)

# Save the cleaned text back into the file
with open('austen_cleaned.txt', 'w', encoding='utf-8') as file:
    file.write(text)