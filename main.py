import os
import time
from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
import re
import json
from flask_cors import CORS
from openai import OpenAI
from youtube_transcript_api.proxies import WebshareProxyConfig

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter

# Get keys from environment
openai_api_key = os.environ.get("OPENAI_API_KEY")
proxy_username = os.environ.get("PROXY_USERNAME", "ibgvbyfk")
proxy_password = os.environ.get("PROXY_PASSWORD", "lktesui61d4c")

# Initialize OpenAI client (new style)
client = OpenAI(api_key=openai_api_key) if openai_api_key else None


# Retry helper
def retry(func, retries=3, delay=2, backoff=2, exceptions=(Exception,), *args, **kwargs):
    """
    Retry a function with exponential backoff.
    :param func: function to call
    :param retries: number of retries
    :param delay: initial delay in seconds
    :param backoff: multiplier for delay
    :param exceptions: exceptions to catch
    """
    attempt = 0
    while attempt < retries:
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            attempt += 1
            if attempt >= retries:
                raise
            wait_time = delay * (backoff ** (attempt - 1))
            print(f"⚠️ Attempt {attempt} failed: {e}. Retrying in {wait_time} sec...")
            time.sleep(wait_time)


# Function to extract YouTube video ID
def get_video_id(url):
    patterns = [
        r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:watch\?v=)([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# Split long transcript into chunks
def chunk_text(text, max_chunk_size=3000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        if current_size + len(word) > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

@app.route('/generate_qa', methods=['POST'])
def generate_qa():
    try:
        # Check if OpenAI client is available
        if not client:
            return jsonify({'error': 'OpenAI API key not configured'}), 500

        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        url = data.get('url', '').strip()
        count = int(data.get('count', 10))

        if not url:
            return jsonify({'error': 'YouTube URL is required'}), 400

        if count < 1 or count > 50:
            return jsonify({'error': 'Question count must be between 1 and 50'}), 400

        video_id = get_video_id(url)
        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL format'}), 400

        # --- Transcript fetch with retries ---
        def fetch_transcript():
            if proxy_username and proxy_password:
                ytt = YouTubeTranscriptApi(
                    proxy_config=WebshareProxyConfig(
                        proxy_username=proxy_username,
                        proxy_password=proxy_password,
                    )
                )
            else:
                ytt = YouTubeTranscriptApi()

            return ytt.fetch(video_id, languages=['en', 'bn', 'hi'])

        try:
            transcript = retry(fetch_transcript, retries=3, delay=2, backoff=2)
        except Exception as e:
            return jsonify({'error': f"Could not fetch transcript after retries: {str(e)}"}), 500

        full_text = " ".join([entry.text for entry in transcript])

        if len(full_text.strip()) < 100:
            return jsonify({'error': 'Transcript too short to generate meaningful questions'}), 400

        chunks = chunk_text(full_text)
        text_to_process = chunks[0]

        # Prompt to OpenAI
        prompt = f"""Based on the following transcript, generate exactly {count} educational question-answer pairs in JSON format.

                    Requirements:
                    • Return ONLY a valid JSON array
                    • Each question should be clear and educational
                    • Each answer should be concise but complete
                    • Focus on key concepts and important information

                    Format:
                    [
                    {{"question": "What is...?", "answer": "The answer is..."}},
                    {{"question": "How does...?", "answer": "It works by..."}}
                    ]

                    Transcript:
                    {text_to_process}"""

        # --- OpenAI call with retries ---
        def call_openai():
            return client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an educational content generator. Always return valid JSON arrays only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )

        try:
            response = retry(call_openai, retries=3, delay=3, backoff=2)
        except Exception as e:
            return jsonify({'error': f"OpenAI request failed after retries: {str(e)}"}), 500

        result = response.choices[0].message.content.strip()

        try:
            json_result = json.loads(result)
            if isinstance(json_result, list):
                return jsonify({'result': result, 'count': len(json_result)})
            else:
                return jsonify({'error': 'Invalid response format from AI'}), 500
        except json.JSONDecodeError:
            return jsonify({'error': 'AI returned invalid JSON format'}), 500

    except Exception as e:
        return jsonify({'error': f"Server error: {str(e)}"}), 500


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'Server is running',
        'openai_configured': bool(openai_api_key),
        'proxy_configured': bool(proxy_username and proxy_password),
        'environment': os.environ.get('FLASK_ENV', 'production')
    })

if __name__ == '__main__': 
    port = int(os.environ.get('PORT', 5500))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'

    if not openai_api_key:
        print("⚠️ Warning: OPENAI_API_KEY not set!")
    
    if not proxy_username or not proxy_password:
        print("⚠️ Warning: Proxy credentials not set, using default values!")

    app.run(host='0.0.0.0', port=port, debug=debug_mode)