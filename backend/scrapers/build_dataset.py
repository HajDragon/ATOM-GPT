import requests
from bs4 import BeautifulSoup
import time
import os
import re

BASE_URL = "http://www.darklyrics.com"
OUTPUT_FILE = r"Z:\GIthub Raps\nanoGPT\backend\data\DarkLyrics\all_lyrics.txt"

# Session with headers
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})

def get_soup(url):
    try:
        print(f"      🌐 Fetching: {url}")
        res = session.get(url, timeout=15)
        res.raise_for_status()
        return BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        print(f"      ❌ Error fetching {url}: {e}")
        return None

def is_valid_band_name(name):
    invalid_names = [
        'SUBMIT LYRICS', 'LINKS', 'PRIVACY', 'CONTACT', 'DISCLAIMER',
        'METAL LYRICS', 'BROWSE', 'SEARCH'
    ]
    
    name = name.upper().strip()
    if not name or len(name) < 2:
        return False
    if any(invalid in name for invalid in invalid_names):
        return False
    return True

def extract_song_lyrics(song_url):
    """Extract actual lyrics from a song page"""
    soup = get_soup(song_url)
    if not soup:
        return None
    
    # DarkLyrics stores lyrics in specific div structures
    # Look for the main content area containing lyrics
    
    # Method 1: Look for divs that contain substantial text content
    for div in soup.find_all('div'):
        div_text = div.get_text().strip()
        
        # Skip if it's clearly navigation or metadata
        if any(skip in div_text.upper() for skip in [
            'SUBMIT LYRICS', 'BROWSE BY BAND', 'DARKLYRICS.COM',
            'PRIVACY', 'DISCLAIMER', 'CONTACT', 'SEARCH LYRICS'
        ]):
            continue
        
        # Look for lyrics pattern: substantial text with line breaks
        if (len(div_text) > 200 and 
            div_text.count('\n') > 8 and
            not div_text.upper().startswith('ALBUM:') and
            'lyrics' not in div_text[:100].lower()):
            
            # Clean the lyrics
            lines = div_text.split('\n')
            clean_lines = []
            
            for line in lines:
                line = line.strip()
                if (line and 
                    not any(nav in line.upper() for nav in [
                        'HTTP://', 'WWW.', 'DARKLYRICS', 'SUBMIT LYRICS',
                        'BROWSE', 'SEARCH', 'CONTACT', 'PRIVACY'
                    ]) and
                    not re.match(r'^\d+\.\s', line) and  # Skip track numbers
                    len(line) > 1):
                    clean_lines.append(line)
            
            if len(clean_lines) > 10:
                return '\n'.join(clean_lines)
    
    # Method 2: Look in the raw text for lyrical content
    page_text = soup.get_text()
    
    # Split into sections and find the largest text block that looks like lyrics
    sections = page_text.split('\n\n')
    best_section = None
    best_score = 0
    
    for section in sections:
        if len(section) > 100:
            lines = [line.strip() for line in section.split('\n') if line.strip()]
            
            # Score based on length and structure
            score = len(lines)
            
            # Reduce score for navigation content
            nav_penalty = sum(1 for line in lines if any(nav in line.upper() for nav in [
                'SUBMIT', 'BROWSE', 'SEARCH', 'HTTP://', 'WWW.'
            ]))
            
            score -= nav_penalty * 3
            
            if score > best_score and score > 10:
                best_section = section
                best_score = score
    
    if best_section:
        # Clean up the best section
        lines = [line.strip() for line in best_section.split('\n') if line.strip()]
        clean_lines = []
        
        for line in lines:
            if (not any(skip in line.upper() for skip in [
                'SUBMIT LYRICS', 'DARKLYRICS', 'BROWSE', 'SEARCH',
                'HTTP://', 'WWW.', 'CONTACT', 'PRIVACY'
            ]) and len(line) > 1):
                clean_lines.append(line)
        
        if len(clean_lines) > 8:
            return '\n'.join(clean_lines)
    
    return None

def construct_lyric_urls(band_soup, band_name):
    """Extract song URLs directly from the band page links"""
    
    song_urls = []
    
    # Method 1: Extract direct song links from the band page
    for link in band_soup.find_all('a', href=True):
        href = link.get('href')
        song_title = link.get_text().strip()
        
        if href and 'lyrics/' in href and '#' in href:
            # Fix the relative URL
            if href.startswith('../'):
                full_url = BASE_URL + href[2:]  # Remove the .. and add base URL
            elif href.startswith('/'):
                full_url = BASE_URL + href
            else:
                full_url = BASE_URL + '/' + href
            
            # Extract album name from URL for organization
            album_match = re.search(r'/lyrics/[^/]+/([^/]+)\.html', href)
            album_name = album_match.group(1) if album_match else "Unknown Album"
            
            song_urls.append((song_title, full_url, album_name))
    
    return song_urls

def extract_song_lyrics_from_anchor(song_url):
    """Extract specific song lyrics from DarkLyrics page using proper HTML parsing"""
    # Remove the anchor to get the base page
    base_url = song_url.split('#')[0]
    anchor = song_url.split('#')[1] if '#' in song_url else '1'
    song_number = int(anchor)
    
    soup = get_soup(base_url)
    if not soup:
        return None
    
    # Get the full page text
    page_text = soup.get_text()
    
    # Split into lines for easier processing
    lines = page_text.split('\n')
    
    # Find all occurrences of this song number
    song_occurrences = []
    for i, line in enumerate(lines):
        line = line.strip()
        if re.match(rf'^{song_number}\.\s*(.+)', line):
            song_occurrences.append(i)
    
    # Use the second occurrence (lyrics section) if available, otherwise first
    if len(song_occurrences) >= 2:
        song_start = song_occurrences[1]  # Second occurrence has the lyrics
    elif len(song_occurrences) == 1:
        song_start = song_occurrences[0]
    else:
        return None
    
    # Find the end of this song's lyrics
    song_end = None
    for i in range(song_start + 1, len(lines)):
        line = lines[i].strip()
        
        # Check if this is the next song number
        if re.match(r'^(\d+)\.\s*(.+)', line):
            next_song_num = int(re.match(r'^(\d+)\.\s*(.+)', line).group(1))
            if next_song_num > song_number:
                song_end = i
                break
        # Check for album boundary or other section markers
        elif (line.startswith('Count ') or 
              'Submits, comments' in line or
              'webmaster@darklyrics.com' in line or
              'Browse by band' in line):
            song_end = i
            break
    
    # If no clear end found, take a reasonable chunk
    if song_end is None:
        song_end = min(song_start + 30, len(lines))
    
    # Extract the lyrics from the song section
    song_lines = lines[song_start + 1:song_end]  # Skip the title line
    
    clean_lyrics = []
    for line in song_lines:
        line = line.strip()
        if (line and 
            len(line) > 0 and
            not re.match(r'^\d+\.\s', line) and  # Skip other song titles
            not 'album:' in line.lower() and
            not line == '[Instrumental]' and  # Skip instrumental markers
            not any(skip in line.upper() for skip in [
                'SUBMIT LYRICS', 'BROWSE BY BAND', 'DARKLYRICS',
                'SEARCH LYRICS', 'CONTACT', 'PRIVACY', 'COPYRIGHT',
                'HTTP://', 'WWW.', 'WEBMASTER@', 'TRANSLATE', 'EMAIL', 'PRINT'
            ])):
            clean_lyrics.append(line)
    
    # Return lyrics if we have substantial content
    if len(clean_lyrics) > 1:
        lyrics_text = '\n'.join(clean_lyrics)
        # Filter out very short "lyrics" that are likely just metadata
        if len(lyrics_text) > 20:
            return lyrics_text
    
    return None

def process_band_page(band_soup, band_name):
    """Enhanced band processing using DarkLyrics URL structure"""
    
    # Get constructed song URLs using the proper DarkLyrics pattern
    song_urls = construct_lyric_urls(band_soup, band_name)
    
    if not song_urls:
        print(f"    ❌ No songs found for {band_name}")
        return None
    
    print(f"    🔍 Found {len(song_urls)} songs, collecting lyrics...")
    
    # Debug: show first few URLs being constructed
    print(f"    🔧 Sample URLs:")
    for i, (song_title, song_url, album) in enumerate(song_urls[:3]):
        print(f"      {i+1}. {song_url}")
    
    result_lines = []
    lyrics_found = 0
    current_album = None
    
    # Process each song URL (increased from 15 to all available)
    for i, (song_title, song_url, album) in enumerate(song_urls):  # Process ALL songs (removed limit)
        # Add album header if new album
        if album != current_album:
            result_lines.append(f"\n=== {album} ===")
            current_album = album
        
        print(f"      🎵 {i+1}: {song_title}")
        
        # Try to get lyrics from the constructed URL
        lyrics = extract_song_lyrics_from_anchor(song_url)
        
        if lyrics and len(lyrics.strip()) > 30:
            result_lines.append(f"\n--- {song_title} ---")
            result_lines.append(lyrics.strip())
            result_lines.append("")
            lyrics_found += 1
            print(f"        ✅ Got lyrics! ({len(lyrics.split())} words)")
        else:
            # Still include the song title even if no lyrics found
            result_lines.append(f"\n--- {song_title} ---")
            result_lines.append("[Lyrics not available or could not be extracted]")
            result_lines.append("")
            print(f"        ❌ No lyrics found")
        
        # Stop if we've found enough lyrics for this band (increased limit)
        if lyrics_found >= 20:  # 20 songs with lyrics per band for comprehensive coverage
            print(f"      ⏹️ Stopping after finding {lyrics_found} songs with lyrics")
            break
    
    print(f"    📊 Successfully collected lyrics for {lyrics_found}/{len(song_urls)} songs")
    
    return '\n'.join(result_lines) if result_lines else None

# Clear old file and start fresh (or resume if file exists)
resume_mode = False
if os.path.exists(OUTPUT_FILE):
    response = input(f"Output file already exists. (R)esume or (S)tart fresh? [R/S]: ").strip().upper()
    if response == 'R':
        resume_mode = True
        print("📂 Resuming from existing file...")
    else:
        os.remove(OUTPUT_FILE)
        print("🗑️  Starting fresh...")
else:
    print("🆕 Creating new dataset file...")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

print("🎵 Metal Lyrics Collector - COMPLETE SITE SCRAPER (SPEED OPTIMIZED)")
print("⚠️  WARNING: This will scrape the ENTIRE DarkLyrics website")
print("📊 Expected: ~4500+ bands across all letters")
print("⚡ SPEED MODE: No delays - Maximum scraping speed")
print("=" * 60)

# Process ALL letters for complete site coverage
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0-9']  # Complete alphabet + numbers

with open(OUTPUT_FILE, "a" if resume_mode else "w", encoding="utf-8") as out:
    if not resume_mode:
        out.write("# Metal Music Dataset with Complete Lyrics\n")
        out.write("# Format: Band Name → Song Titles → Full Lyrics\n")
        out.write("# For personal AI training\n\n")
    
    total_bands = 0
    total_songs_with_lyrics = 0
    
    # If resuming, count existing bands
    if resume_mode:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as check_file:
            content = check_file.read()
            total_bands = content.count("### ") 
            total_songs_with_lyrics = content.count("✅ Got lyrics!")
            print(f"📊 Resume stats: {total_bands} bands, {total_songs_with_lyrics} songs already collected")
    
    for letter_idx, letter in enumerate(letters):
        letter_url = f"{BASE_URL}/{letter}.html"
        print(f"\n🔍 Processing letter '{letter.upper()}' ({letter_idx + 1}/{len(letters)})...")
        
        soup = get_soup(letter_url)
        if not soup:
            continue
        
        # Get band links
        band_links = []
        for link in soup.find_all("a", href=True):
            href = link.get("href")
            name = link.text.strip()
            
            if (href and href.endswith(".html") and 
                "/" in href and "lyrics/" not in href and
                is_valid_band_name(name)):
                
                band_links.append((name, BASE_URL + "/" + href))
        
        print(f"  📀 Found {len(band_links)} bands")
        
        # Process ALL bands for complete dataset coverage
        for i, (band_name, band_url) in enumerate(band_links):  # Process ALL bands (removed limit)
            print(f"  🎸 Processing band {i+1}/{len(band_links)}: {band_name}")
            
            try:
                band_soup = get_soup(band_url)
                if band_soup:
                    lyrics_content = process_band_page(band_soup, band_name)
                    
                    if lyrics_content:
                        out.write(f"\n### {band_name} ###\n")
                        out.write(lyrics_content)
                        out.write("\n" + "="*50 + "\n")
                        total_bands += 1
                        
                        # Count songs with lyrics from this band
                        songs_count = lyrics_content.count("✅ Got lyrics!")
                        total_songs_with_lyrics += songs_count
                        
                        print(f"    ✅ Added {band_name} ({songs_count} songs with lyrics)")
                        
                        # Periodic progress update
                        if total_bands % 50 == 0:
                            print(f"\n📊 PROGRESS UPDATE: {total_bands} bands processed, {total_songs_with_lyrics} songs with lyrics collected")
                            out.flush()  # Save progress to file
                    else:
                        print(f"    ❌ No lyrics found for {band_name}")
                else:
                    print(f"    ❌ Failed to fetch page for {band_name}")
            
            except Exception as e:
                print(f"    💥 Error processing {band_name}: {e}")
                continue  # Skip this band and continue with the next one

print(f"\n🎉 COMPLETE! Successfully processed {total_bands} bands from the entire DarkLyrics website")
print(f"📈 Total songs with lyrics collected: {total_songs_with_lyrics}")
print(f"💾 Output saved to: {OUTPUT_FILE}")
print(f"📁 File size: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.1f} MB")
