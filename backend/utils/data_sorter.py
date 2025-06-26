#!/usr/bin/env python3
"""
DarkLyrics Data Sorter for AI Training
=====================================

This script transforms the raw DarkLyrics dataset into a structured format
that's more suitable for AI training by clearly labeling the context hierarchy:
Band -> Album -> Song -> Lyrics

Input format:
### BAND NAME ###
=== album name ===
--- song title ---
lyrics content...

Output format:
<Band>BAND NAME</Band>
<Album>album name</Album>
<Song>song title</Song>
<Lyrics>
lyrics content...
</Lyrics>

This structured format helps the AI model understand the hierarchical relationships
and context of the musical content.
"""

import re
import os
from typing import Optional, List, Tuple

class DarkLyricsDataSorter:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.stats = {
            'bands_processed': 0,
            'albums_processed': 0,
            'songs_processed': 0,
            'lines_processed': 0,
            'lyrics_blocks': 0
        }
    
    def parse_and_structure_data(self):
        """Main method to parse the input file and create structured output"""
        print("🎵 Starting DarkLyrics Data Structuring...")
        print(f"📂 Input: {self.input_file}")
        print(f"💾 Output: {self.output_file}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        current_band = None
        current_album = None
        current_song = None
        lyrics_buffer = []
        
        with open(self.input_file, 'r', encoding='utf-8') as infile, \
             open(self.output_file, 'w', encoding='utf-8') as outfile:
            
            # Write header
            outfile.write("# Structured Metal Music Dataset for AI Training\n")
            outfile.write("# Format: <Band>name</Band> -> <Album>name</Album> -> <Song>name</Song> -> <Lyrics>content</Lyrics>\n")
            outfile.write("# Source: DarkLyrics.com\n\n")
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                self.stats['lines_processed'] = line_num
                
                # Progress indicator
                if line_num % 10000 == 0:
                    print(f"📊 Processed {line_num:,} lines, {self.stats['bands_processed']} bands, {self.stats['songs_processed']} songs")
                
                # Skip header comments
                if line.startswith('#') and line_num < 10:
                    continue
                
                # Check for band name: ### BAND NAME ###
                band_match = re.match(r'^### (.+) ###$', line)
                if band_match:
                    # Write any pending lyrics first
                    if current_song and lyrics_buffer:
                        self._write_lyrics_block(outfile, lyrics_buffer)
                        lyrics_buffer = []
                    
                    current_band = band_match.group(1).strip()
                    current_album = None
                    current_song = None
                    
                    outfile.write(f"\n<Band>{current_band}</Band>\n")
                    self.stats['bands_processed'] += 1
                    continue
                
                # Check for album name: === album name ===
                album_match = re.match(r'^=== (.+) ===$', line)
                if album_match:
                    # Write any pending lyrics first
                    if current_song and lyrics_buffer:
                        self._write_lyrics_block(outfile, lyrics_buffer)
                        lyrics_buffer = []
                    
                    current_album = album_match.group(1).strip()
                    current_song = None
                    
                    outfile.write(f"<Album>{current_album}</Album>\n")
                    self.stats['albums_processed'] += 1
                    continue
                
                # Check for song name: --- song title ---
                song_match = re.match(r'^--- (.+) ---$', line)
                if song_match:
                    # Write any pending lyrics first
                    if current_song and lyrics_buffer:
                        self._write_lyrics_block(outfile, lyrics_buffer)
                        lyrics_buffer = []
                    
                    current_song = song_match.group(1).strip()
                    
                    outfile.write(f"<Song>{current_song}</Song>\n")
                    self.stats['songs_processed'] += 1
                    continue
                
                # Check for empty line (end of lyrics block)
                if not line and lyrics_buffer:
                    self._write_lyrics_block(outfile, lyrics_buffer)
                    lyrics_buffer = []
                    continue
                
                # Check for separator line
                if line.startswith('====='):
                    # Write any pending lyrics first
                    if current_song and lyrics_buffer:
                        self._write_lyrics_block(outfile, lyrics_buffer)
                        lyrics_buffer = []
                    continue
                
                # Collect lyrics content
                if line and current_song:
                    lyrics_buffer.append(line)
            
            # Write any remaining lyrics
            if current_song and lyrics_buffer:
                self._write_lyrics_block(outfile, lyrics_buffer)
        
        self._print_final_stats()
    
    def _write_lyrics_block(self, outfile, lyrics_buffer: List[str]):
        """Write a structured lyrics block to the output file"""
        if not lyrics_buffer:
            return
        
        outfile.write("<Lyrics>\n")
        for lyric_line in lyrics_buffer:
            outfile.write(f"{lyric_line}\n")
        outfile.write("</Lyrics>\n\n")
        
        self.stats['lyrics_blocks'] += 1
    
    def _print_final_stats(self):
        """Print final processing statistics"""
        print("\n🎉 Data structuring complete!")
        print("=" * 50)
        print(f"📈 Processing Statistics:")
        print(f"  • Lines processed: {self.stats['lines_processed']:,}")
        print(f"  • Bands processed: {self.stats['bands_processed']:,}")
        print(f"  • Albums processed: {self.stats['albums_processed']:,}")
        print(f"  • Songs processed: {self.stats['songs_processed']:,}")
        print(f"  • Lyrics blocks: {self.stats['lyrics_blocks']:,}")
        print(f"💾 Output saved to: {self.output_file}")
        
        # Calculate file size
        if os.path.exists(self.output_file):
            file_size = os.path.getsize(self.output_file) / (1024 * 1024)
            print(f"📁 Output file size: {file_size:.1f} MB")


class AlternativeStructuredFormatter:
    """Alternative formatter for different AI training approaches"""
    
    @staticmethod
    def create_training_format(input_file: str, output_file: str):
        """Create a format optimized for language model training"""
        print("🤖 Creating AI training optimized format...")
        
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            current_context = {}
            
            for line in infile:
                line = line.strip()
                
                # Parse band
                if line.startswith('<Band>') and line.endswith('</Band>'):
                    current_context['band'] = line[6:-7]
                
                # Parse album
                elif line.startswith('<Album>') and line.endswith('</Album>'):
                    current_context['album'] = line[7:-8]
                
                # Parse song
                elif line.startswith('<Song>') and line.endswith('</Song>'):
                    current_context['song'] = line[6:-7]
                
                # Parse lyrics
                elif line == '<Lyrics>':
                    lyrics = []
                    # Read until </Lyrics>
                    for next_line in infile:
                        next_line = next_line.strip()
                        if next_line == '</Lyrics>':
                            break
                        lyrics.append(next_line)
                    
                    if lyrics and all(key in current_context for key in ['band', 'album', 'song']):
                        # Write training format
                        outfile.write(f"Band: {current_context['band']} | ")
                        outfile.write(f"Album: {current_context['album']} | ")
                        outfile.write(f"Song: {current_context['song']}\n")
                        outfile.write("Lyrics:\n")
                        for lyric_line in lyrics:
                            outfile.write(f"{lyric_line}\n")
                        outfile.write("\n---\n\n")


def main():
    """Main execution function"""
    # File paths
    input_file = r"Z:\GIthub Raps\nanoGPT\backend\data\DarkLyrics\all_lyrics.txt"
    structured_output = r"Z:\GIthub Raps\nanoGPT\backend\data\DarkLyrics\structured_lyrics.txt"
    training_output = r"Z:\GIthub Raps\nanoGPT\backend\data\DarkLyrics\training_formatted_lyrics.txt"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        return
    
    print("🎵 DarkLyrics Data Structuring Tool")
    print("=" * 50)
    
    # Option 1: Create structured XML-like format
    sorter = DarkLyricsDataSorter(input_file, structured_output)
    sorter.parse_and_structure_data()
    
    print("\n" + "=" * 50)
    
    # Option 2: Create alternative training format
    print("🤖 Creating alternative training format...")
    AlternativeStructuredFormatter.create_training_format(structured_output, training_output)
    print(f"✅ Training format saved to: {training_output}")


if __name__ == "__main__":
    main()
