from flask import Flask, render_template, request, jsonify, send_file
import os
from pipeline import generate_script_from_wikipedia, generate_mp3_from_script
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/audio'

# Ensure audio directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Read prompt template
PROMPT_TEMPLATE = None
if os.path.exists('prompt.txt'):
    with open('prompt.txt', 'r') as f:
        PROMPT_TEMPLATE = f.read()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate-script', methods=['POST'])
def generate_script():
    """Generate script from Wikipedia URL (step 1)."""
    try:
        data = request.json
        wiki_url = data.get('wiki_url', '').strip()
        
        if not wiki_url:
            return jsonify({'error': 'Wikipedia URL is required'}), 400
        
        # Generate script only
        script = generate_script_from_wikipedia(wiki_url, PROMPT_TEMPLATE)
        
        return jsonify({
            'success': True,
            'script': script
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate-mp3', methods=['POST'])
def generate_mp3():
    """Generate MP3 from script (step 2)."""
    try:
        data = request.json
        script = data.get('script', '').strip()
        
        if not script:
            return jsonify({'error': 'Script is required'}), 400
        
        # Generate MP3 from script
        final_audio = generate_mp3_from_script(script)
        
        # Save audio to file
        audio_filename = 'generated_audio.mp3'
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        final_audio.export(audio_path, format="mp3")
        
        # Get audio duration in seconds
        duration_seconds = len(final_audio) / 1000.0
        
        return jsonify({
            'success': True,
            'audio_url': f'/static/audio/{audio_filename}',
            'duration': duration_seconds
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/static/audio/<filename>')
def serve_audio(filename):
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype='audio/mpeg')
    return jsonify({'error': 'Audio file not found'}), 404


if __name__ == '__main__':
    app.run(debug=True, port=5000)

