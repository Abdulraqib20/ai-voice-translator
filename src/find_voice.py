import requests
import json
import sys
import os

# Add the parent directory to the system path to import your API key from the appconfig file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.appconfig import ELEVENLABS_API_KEY, VOICE_ID

# Define the API endpoint for fetching voices
url = "https://api.elevenlabs.io/v1/voices"

# Set up the headers with the API key for authentication
headers = {
    "Accept": "application/json",
    "xi-api-key": ELEVENLABS_API_KEY,
    "Content-Type": "application/json"
}

# Send a GET request to fetch the voices
response = requests.get(url, headers=headers)

# Parse the JSON response
if response.status_code == 200:
    data = response.json()
    
    # Check if 'voices' key is present in the response
    if 'voices' in data:
        # Iterate through the voices and print their details
        print("Available voices:")
        found_voice = False
        for voice in data['voices']:
            print(f"{voice['name']}; {voice['voice_id']}")
            # Check for the specific voice name or ID
            if voice['name'] == "Exhibit 1" or voice['voice_id'] == VOICE_ID:
                found_voice = True
        
        # If the voice is not found, notify the user
        if not found_voice:
            print(f"\nThe voice 'Exhibit 1' with ID {VOICE_ID} was not found. Ensure it is active on the account.")
    else:
        print("Error: 'voices' key not found in the API response.")
        print("Full response:", json.dumps(data, indent=2))
else:
    # Print an error message if the request fails
    print(f"Failed to fetch voices. HTTP Status Code: {response.status_code}")
    print("Response:", response.text)
