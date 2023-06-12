

import csv
import re
from collections import Counter
from PyPDF2 import PdfFileReader

# Set the path to your PDF file
pdf_path = 'PATH_TO_PDF_FILE'

# Read the PDF file
with open(pdf_path, 'rb') as file:
    pdf = PdfFileReader(file)

    # Extract text from each page
    text = ''
    for page in range(pdf.getNumPages()):
        text += pdf.getPage(page).extractText()

# Preprocess the extracted text
text = re.sub('\s+', ' ', text).strip()

# Split the text into words
words = re.findall('\w+', text.lower())

# Find the most repeated word
word_counts = Counter(words)
most_repeated_word = word_counts.most_common(1)[0][0]

# Prepare the CSV file
csv_file = open('pdf_text.csv', 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['Text'])

# Write the extracted text to the CSV file
writer.writerow([text])

csv_file.close()

print("Most repeated word:", most_repeated_word)
