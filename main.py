import os

from flask import Flask, request, jsonify
from text_segmenter import MiniSegTextSegmenter
from video_segmenter import VideoSegmenter
from openai_summary_generator import OpenAiSummaryGenerator

app = Flask(__name__)
api_key = os.environ["OPENAI_API_KEY"]

text_segmenter = MiniSegTextSegmenter("./pretrained")
summary_generator = OpenAiSummaryGenerator(api_key=api_key)
video_segmenter = VideoSegmenter(text_segmenter, summary_generator)


@app.route("/segment/text", methods=["POST"])
def segment_text():
    text_data = request.get_json()
    text = text_data.get("text", [])

    if not text:
        return "No text provided", 400

    threshold = request.args.get("threshold", type=float)
    segments = text_segmenter.segment_text(text, threshold)

    result = {"segments": segments}

    generate_summaries = text_data.get("generate_summaries", False)
    if generate_summaries:
        summaries = summary_generator.generate_summaries(segments)
        result["summaries"] = [summary.model_dump_json() for summary in summaries]

    return jsonify(result)


@app.route("/segment/yt", methods=["POST"])
def segment_yt():
    video_id = request.args.get("video_id")

    if video_id is None:
        return "Couldn't load video", 400

    threshold = request.args.get("threshold", type=float)
    generate_summaries = request.args.get("generate_summaries", False, type=bool)

    segmenter_res = video_segmenter.segment_video(
        video_id, generate_summaries, threshold
    )

    if generate_summaries:
        timestamps, summaries = segmenter_res
        result = {
            "timestamps": timestamps,
            "summaries": [summary.model_dump_json() for summary in summaries],
        }
    else:
        timestamps = segmenter_res
        result = {"timestamps": timestamps}

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("TEXT_SEGMENTER_PORT", 5000))
    app.run(host="0.0.0.0", port=port)
