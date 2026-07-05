"""
Scrapes Helldivers 2 Steam reviews using Valve's official public appreviews
endpoint (no login, no API key required):
https://partner.steamgames.com/doc/store/getreviews

Compliant usage notes:
- Rate-limited to 1 request/sec (well below what the endpoint can handle)
- Identifies itself via a descriptive User-Agent (good practice, not required)
- Pulls up to MAX_REVIEWS reviews then stops (no indefinite hammering)
"""

import requests
import pandas as pd
import time

APP_ID = 553850  # Helldivers 2
MAX_REVIEWS = 5000  # adjust as needed for your capstone scope
OUTPUT_PATH = 'data/helldivers2_reviews.csv'

HEADERS = {
    'User-Agent': 'CapstoneToxicityClassifier/1.0 (educational research project)'
}

def fetch_reviews(app_id, max_reviews=5000):
    reviews = []
    cursor = '*'  # required starting value for the first request
    seen_cursors = set()

    while len(reviews) < max_reviews:
        params = {
            'json': 1,
            'filter': 'all',
            'language': 'english',
            'review_type': 'all',
            'purchase_type': 'all',
            'num_per_page': 100,
            'cursor': cursor,
        }

        response = requests.get(
            f'https://store.steampowered.com/appreviews/{app_id}',
            params=params,
            headers=HEADERS,
        )
        response.raise_for_status()
        data = response.json()

        batch = data.get('reviews', [])
        if not batch:
            print("No more reviews returned. Stopping.")
            break

        for r in batch:
            reviews.append({
                'review_id': r.get('recommendationid'),
                'text': r.get('review'),
                'voted_up': r.get('voted_up'),
                'votes_up': r.get('votes_up'),
                'votes_funny': r.get('votes_funny'),
                'playtime_at_review_minutes': r.get('author', {}).get('playtime_at_review'),
                'timestamp_created': r.get('timestamp_created'),
                'language': r.get('language'),
            })

        print(f"Fetched {len(reviews)} reviews so far...")

        new_cursor = data.get('cursor')
        if not new_cursor or new_cursor in seen_cursors:
            print("Reached end of available reviews. Stopping.")
            break
        seen_cursors.add(new_cursor)
        cursor = new_cursor

        time.sleep(1)  # be polite — 1 request/sec

    return reviews[:max_reviews]

if __name__ == '__main__':
    print(f"Scraping up to {MAX_REVIEWS} reviews for app {APP_ID}...")
    reviews = fetch_reviews(APP_ID, MAX_REVIEWS)
    df = pd.DataFrame(reviews)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone! Saved {len(df)} reviews to {OUTPUT_PATH}")