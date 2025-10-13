# generate_youtube_links.py
import json
from youtubesearchpython import VideosSearch
from skills_db import SKILLS_DB
import time

def generate_links():
    """
    Iterates through our master skill list, finds a relevant YouTube tutorial
    for each, and saves the results to a JSON file.
    """
    youtube_links_map = {}
    print(f"🚀 Starting to generate YouTube links for {len(SKILLS_DB)} skills...")

    for i, skill in enumerate(SKILLS_DB):
        # To get better results, we formulate a more specific search query.
        search_query = f"{skill} tutorial for beginners"
        
        try:
            # We only need the top result.
            search = VideosSearch(search_query, limit=1)
            results = search.result()['result']
            
            if results:
                top_result = results[0]
                video_title = top_result['title']
                video_link = top_result['link']
                youtube_links_map[skill] = {
                    "title": video_title,
                    "link": video_link
                }
                print(f"({i+1}/{len(SKILLS_DB)}) ✅ Found '{video_title}' for '{skill}'")
            else:
                youtube_links_map[skill] = None
                print(f"({i+1}/{len(SKILLS_DB)}) ❌ No results found for '{skill}'")

        except Exception as e:
            print(f"({i+1}/{len(SKILLS_DB)}) ❗ Error searching for '{skill}': {e}")
            youtube_links_map[skill] = None
        
        # Be respectful to YouTube's servers by adding a small delay.
        time.sleep(1) 

    output_filename = 'youtube_links.json'
    print(f"\n💾 Saving all links to '{output_filename}'...")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(youtube_links_map, f, indent=4)

    print("🎉 Link generation complete!")

if __name__ == '__main__':
    generate_links()