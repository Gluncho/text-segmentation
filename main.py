import os

from flask import Flask, request, jsonify
from text_segmenter import MiniSegTextSegmenter
from video_segmenter import VideoSegmenter

app = Flask(__name__)

segmenter = MiniSegTextSegmenter('./pretrained')


@app.route('/segment/text', methods=['POST'])
def segment_text():
    text_data = request.get_json()
    text = text_data.get('text', [])

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    segments = segmenter.segment_text(text)

    return jsonify({'segments': segments})


@app.route('/segment/yt', methods=['POST'])
def segment_yt():
    video_id = request.args.get('video_id')
    video_segmenter = VideoSegmenter(MiniSegTextSegmenter('./pretrained'))
    timestamps = video_segmenter.segment_video(video_id)
    return jsonify({'timestamps': timestamps})


if __name__ == '__main__':
    port = int(os.environ.get('TEXT_SEGMENTER_PORT', 5000))
    app.run(host='0.0.0.0', port=port)
