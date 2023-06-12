


import csv
from googleapiclient.discovery import build

# Set your API key
API_KEY = 'YOUR_API_KEY'

# Set the YouTube video ID or URL
video_id = 'VIDEO_ID_OR_URL'

# Build the YouTube Data API client
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Request the video comments
response = youtube.commentThreads().list(
    part='snippet',
    videoId=video_id,
    maxResults=100,  
    order='relevance'  
).execute()

# Prepare the CSV file
csv_file = open('video_comments.csv', 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['Comment', 'Like Count'])

# Extract comments and save them to the CSV file
for item in response['items']:
    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
    like_count = item['snippet']['topLevelComment']['snippet']['likeCount']
    writer.writerow([comment, like_count])

csv_file.close()
