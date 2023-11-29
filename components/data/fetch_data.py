import requests
import json
import datetime
import os

def datetime_to_unix(dt):
    """Convert datetime to Unix timestamp in milliseconds."""
    return int(dt.timestamp() * 1000)

def save_data(new_data):
    """Save unique data to a file."""
    if os.path.exists("data.json"):
        with open("data.json", "r+") as file:
            file_data = json.load(file)
            existing_timestamps = {item['date'] for item in file_data}

            # Add only new unique data
            for item in new_data['history']:
                if item['date'] not in existing_timestamps:
                    file_data.append(item)

            file.seek(0)  # Reset file position to the beginning
            json.dump(file_data, file)
    else:
        with open("data.json", "w") as file:
            json.dump(new_data['history'], file)

def fetch_data(start_date, stop_date):
    """Fetch data from API between start_date and stop_date."""
    # API URL and your API key
    url = "https://api.livecoinwatch.com/coins/single/history"
    api_key = "8c27dfd2-188e-46ef-9207-647c442d5702"

    # Headers for API request
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': api_key
    }

    # Initialize start_datetime and stop_datetime
    start_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M")
    stop_datetime = datetime.datetime.strptime(stop_date, "%Y-%m-%d %H:%M")

    while start_datetime > stop_datetime:
        # Calculate end and start timestamps
        end_timestamp = datetime_to_unix(start_datetime)
        start_datetime -= datetime.timedelta(hours=12)
        start_timestamp = datetime_to_unix(start_datetime)
        date_key = start_datetime.strftime("%Y-%m-%d %H:%M")

        # Prepare payload
        payload = json.dumps({
            "currency": "USD",
            "code": "BTC",
            "start": start_timestamp,
            "end": end_timestamp
        })

        try:
            # Making the API request
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()

            # Save response data under the date key
            save_data(response.json())

            # Log success
            print(f"Data saved for period: {start_datetime} to {start_datetime + datetime.timedelta(hours=12)}")

        except requests.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            break
        except requests.RequestException as e:
            print(f"Error during requests to {url}: {e}")
            break

if __name__ == "__main__":
    fetch_data("2018-10-06 22:00", "2015-01-01 00:00")
