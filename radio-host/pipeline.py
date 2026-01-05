import requests
import os
import re
import io
import random
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from pydub import AudioSegment
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Setting voice for speakers
VOICE_MAP = {
    "speaker1": "3AMU7jXQuQa3oRvRqUmb",  # slightly deeper
    "speaker2": "OtEfb2LVzIE45wdYe54M"   # slightly lighter 
}


def fetch_wikipedia_content(wiki_url: str):
    """Fetch and parse Wikipedia content."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'
    }
    
    response = requests.get(wiki_url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return soup


def extract_headings(soup):
    """Extract the top H1 & H2 headings."""
    headings = []
    
    # H1 (page title)
    h1 = soup.find("h1", {"id": "firstHeading"})
    if h1:
        headings.append(("H1", h1.get_text(strip=True)))
    
    # H2 headings
    h2_tags = soup.find_all("h2")
    for h2 in h2_tags:
        if len(headings) >= 3:
            break
        
        text = h2.get_text(strip=True)
        if text and text != "Contents":
            headings.append(("H2", text))
    
    return headings


def extract_combined_content(soup, headings):
    """Combine paragraphs from the extracted headings."""
    if not headings:
        return ""

    # Find start node (first heading)
    first_level, first_title = headings[0]
    if first_level == "H1":
        start_node = soup.find("h1", string=first_title)
    else:
        start_node = soup.find("h2", string=first_title)

    # Find end node (last heading)
    last_level, last_title = headings[-1]
    if last_level == "H1":
        end_node = soup.find("h1", string=last_title)
    else:
        end_node = soup.find("h2", string=last_title)

    if not start_node or not end_node:
        return ""

    collected = []
    started = False

    for tag in soup.find_all(["h1", "h2", "p"]):
        if tag == start_node:
            started = True
            continue

        if started:
            # Stop when we hit the next heading after the last one
            if tag.name in ["h1", "h2"] and tag != end_node:
                break

            if tag.name == "p":
                text = tag.get_text(strip=True)
                if len(text) > 50:
                    collected.append(text)

    return "\n".join(collected)


def clean_wikipedia_text(text: str) -> str:
    """Clean the text."""
    # Remove citations [1], [20], [k], etc.
    text = re.sub(r"\[[^\]]*\]", "", text)

    # Fix lowercase-uppercase joins: theRepublic -> the Republic
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Fix punctuation followed by letter: India,is -> India, is
    text = re.sub(r"([.,;:])([A-Za-z])", r"\1 \2", text)

    # Fix letter-number joins: basin9 -> basin 9
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)

    # Fix number-letter joins: 1200BCE -> 1200 BCE
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def generate_script_from_content(clean_content: str, prompt_template: str = None) -> str:
    """Uses LLM to generate a conversation script from cleaned Wikipedia content."""
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    if prompt_template:
        PROMPT = f"{prompt_template}\n\nContent:\n{clean_content}"
    else:
        PROMPT = f"Content:\n{clean_content}"

    response = client.chat.completions.create(
        model="gpt-5.2",   # Note: Original had "gpt-5.2" which may not be available
        messages=[
            {"role": "user", "content": PROMPT}
        ],
        temperature=0.6
    )

    return response.choices[0].message.content.strip()


def parse_conversation(script: str):
    """Parses a speaker-labelled transcript into ordered turns."""
    turns = []
    lines = script.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(r"(Speaker\s*[12]):\s*(.+)", line, re.IGNORECASE)
        if not match:
            continue

        speaker_raw, text = match.groups()
        speaker_id = "speaker1" if "1" in speaker_raw else "speaker2"

        cleaned_text = text.strip()
        turns.append((speaker_id, cleaned_text))

    return turns


def tts_turn_elevenlabs(text: str, speaker_id: str) -> AudioSegment:
    """Invoke ElevenLabs TTS for a single turn."""
    if not ELEVENLABS_API_KEY:
        raise Exception("ELEVENLABS_API_KEY not set")
    
    voice_id = VOICE_MAP[speaker_id]

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model_id": "eleven_v3",
        "voice_settings": {
            "stability": 0.5,           # key for conversation
            "similarity_boost": 0.75
        }
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        print("Status:", response.status_code)
        print("Response:", response.text)
        raise Exception("ElevenLabs TTS failed")

    return AudioSegment.from_file(
        io.BytesIO(response.content),
        format="mp3"
    )


def natural_pause_ms():
    """Generate natural pause duration."""
    return random.choice([200, 250, 300, 350, 400])


def stitch_conversation_elevenlabs(turns):
    """Stitch conversation turns with pauses."""
    final_audio = AudioSegment.silent(0)

    for i, (speaker, text) in enumerate(turns):
        segment = tts_turn_elevenlabs(text, speaker)
        final_audio += segment

        if i < len(turns) - 1:
            final_audio += AudioSegment.silent(natural_pause_ms())

    return final_audio


def generate_script_from_wikipedia(wiki_url: str, prompt_template: str = None):
    """Generate script from Wikipedia URL (step 1)."""
    # Fetch and parse Wikipedia
    soup = fetch_wikipedia_content(wiki_url)
    
    # Extract headings
    headings = extract_headings(soup)
    
    # Extract combined content
    combined_content = extract_combined_content(soup, headings)
    
    # Clean content
    clean_content = clean_wikipedia_text(combined_content)
    
    # Generate script
    script = generate_script_from_content(clean_content, prompt_template)
    
    return script


def generate_mp3_from_script(script: str):
    """Generate MP3 from script (step 2)."""
    # Parse conversation
    turns = parse_conversation(script)
    
    # Generate audio
    final_audio = stitch_conversation_elevenlabs(turns)
    
    return final_audio


def process_wikipedia_url(wiki_url: str, prompt_template: str = None):
    """Main pipeline function to process Wikipedia URL and generate MP3 (legacy function)."""
    # Fetch and parse Wikipedia
    soup = fetch_wikipedia_content(wiki_url)
    
    # Extract headings
    headings = extract_headings(soup)
    
    # Extract combined content
    combined_content = extract_combined_content(soup, headings)
    
    # Clean content
    clean_content = clean_wikipedia_text(combined_content)
    
    # Generate script
    script = generate_script_from_content(clean_content, prompt_template)
    
    # Parse conversation
    turns = parse_conversation(script)
    
    # Generate audio
    final_audio = stitch_conversation_elevenlabs(turns)
    
    return script, final_audio

