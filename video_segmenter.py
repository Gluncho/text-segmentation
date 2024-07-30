import nltk.data
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from text_segmenter import MiniSegTextSegmenter

nltk.download('punkt')


class VideoSegmenter:
    def __init__(self, text_segmenter):
        self.text_segmenter = text_segmenter

    def segment_video(self, video_id: str):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            lines = [line['text'] for line in transcript]
            sentences = self._captions_to_sentences(lines)
        except TranscriptsDisabled:
            raise Exception('Could not retrieve transcript for video id: {}'.format(video_id))
        segments = self.text_segmenter.segment_text(sentences)
        return self._generate_timestamps(transcript, segments)

    def _captions_to_sentences(self, captions: list[str]):
        text = ' '.join(captions)
        sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sentence_detector.tokenize(text)
        return sentences

    def _generate_timestamps(self, captions: list[dict], segments: list[str]):
        segment_ending_indices = self._get_segment_endings_indices(segments)
        timestamps = []

        captions_idx = 0
        eaten_so_far = 0

        for segment_ending_index in segment_ending_indices:
            while (eaten_so_far + len(captions[captions_idx]['text'])) < segment_ending_index:
                eaten_so_far += len(captions[captions_idx]['text']) + 1
                captions_idx += 1
            segment_intersection_len = segment_ending_index - eaten_so_far + 1

            split_timestamp = (segment_intersection_len / len(captions[captions_idx]['text'])) * captions[captions_idx][
                'duration'] + captions[captions_idx]['start']
            timestamps.append(split_timestamp)

        return timestamps

    def _get_segment_endings_indices(self, segments: list[str]):
        res = []
        offset = 0
        for segment in segments[:-1]:
            offset += len(segment)
            res.append(offset - 1)

        return res

