import os

from flask import Flask, request, jsonify
from text_segmenter import MiniSegTextSegmenter
from video_segmenter import VideoSegmenter

app = Flask(__name__)

text_segmenter = MiniSegTextSegmenter('./pretrained')
video_segmenter = VideoSegmenter(text_segmenter)


@app.route('/segment/text', methods=['POST'])
def segment_text():
    text_data = request.get_json()
    text = text_data.get('text', [])

    if not text:
        return 'No text provided', 400

    threshold = request.args.get('threshold', type=float)
    segments = text_segmenter.segment_text(text, threshold)

    return jsonify({'segments': segments})


@app.route('/segment/yt', methods=['POST'])
def segment_yt():
    video_id = request.args.get('video_id')

    if video_id is None:
        return "Couldn't load video", 400

    threshold = request.args.get('threshold', type=float)
    timestamps = video_segmenter.segment_video(video_id, threshold)
    return jsonify({'timestamps': timestamps})


if __name__ == '__main__':
    port = int(os.environ.get('TEXT_SEGMENTER_PORT', 5000))
    app.run(host='0.0.0.0', port=port)
