import os

from flask import Flask, request, jsonify
from text_segmenter import MiniSegTextSegmenter

app = Flask(__name__)

segmenter = MiniSegTextSegmenter('./pretrained')


@app.route('/segment', methods=['POST'])
def segment_text():
    text_data = request.get_json()
    text = text_data.get('text', [])

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    segments = segmenter.segment_text(text)

    return jsonify({'segments': segments})


if __name__ == '__main__':
    port = int(os.environ.get('TEXT_SEGMENTER_PORT', 5000))
    app.run(host='0.0.0.0', port=port)
