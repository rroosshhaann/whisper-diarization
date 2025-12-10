"""
Core diarization logic extracted from diarize.py for API use.
Returns Deepgram-compatible response format.
"""
import io
import logging
import os
import re
from typing import Callable
from uuid import uuid4

import faster_whisper
import torch
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel

from helpers import (
    cleanup,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    format_timestamp,
)

MTYPES = {"cpu": "int8", "cuda": "float16"}


def run_diarization(
    audio_path: str,
    model_name: str = "medium.en",
    language: str | None = None,
    stemming: bool = True,
    suppress_numerals: bool = False,
    device: str | None = None,
    batch_size: int = 8,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Run full diarization pipeline on an audio file.

    Args:
        audio_path: Path to audio file
        model_name: Whisper model name (default: medium.en)
        language: Language code or None for auto-detect
        stemming: Whether to separate vocals from music
        suppress_numerals: Convert digits to written text
        device: "cuda" or "cpu" (auto-detect if None)
        batch_size: Batch size for inference
        progress_callback: Optional callback for progress updates

    Returns:
        dict with keys: transcript, srt, segments
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def update_progress(stage: str):
        if progress_callback:
            progress_callback(stage)

    pid = os.getpid()
    temp_outputs_dir = f"temp_outputs_{pid}"

    # Process language argument
    language = process_language_arg(language, model_name)

    # Step 1: Audio preprocessing (optional stem separation)
    update_progress("separating_vocals")

    if stemming:
        return_code = os.system(
            f'python -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "{temp_outputs_dir}" --device "{device}"'
        )

        if return_code != 0:
            logging.warning("Source splitting failed, using original audio file.")
            vocal_target = audio_path
        else:
            vocal_target = os.path.join(
                temp_outputs_dir,
                "htdemucs",
                os.path.splitext(os.path.basename(audio_path))[0],
                "vocals.wav",
            )
    else:
        vocal_target = audio_path

    # Step 2: Transcription
    update_progress("transcribing")

    whisper_model = faster_whisper.WhisperModel(
        model_name, device=device, compute_type=MTYPES[device]
    )
    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
    audio_waveform = faster_whisper.decode_audio(vocal_target)

    suppress_tokens = (
        find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
        if suppress_numerals
        else [-1]
    )

    if batch_size > 0:
        transcript_segments, info = whisper_pipeline.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            batch_size=batch_size,
        )
    else:
        transcript_segments, info = whisper_model.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            vad_filter=True,
        )

    full_transcript = "".join(segment.text for segment in transcript_segments)

    del whisper_model, whisper_pipeline
    torch.cuda.empty_cache()

    # Step 3: Forced alignment
    update_progress("aligning")

    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    emissions, stride = generate_emissions(
        alignment_model,
        torch.from_numpy(audio_waveform)
        .to(alignment_model.dtype)
        .to(alignment_model.device),
        batch_size=batch_size,
    )

    del alignment_model
    torch.cuda.empty_cache()

    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language=langs_to_iso[info.language],
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    # Step 4: Diarization
    update_progress("diarizing")

    from diarization import MSDDDiarizer
    diarizer_model = MSDDDiarizer(device=device)
    speaker_ts = diarizer_model.diarize(torch.from_numpy(audio_waveform).unsqueeze(0))

    del diarizer_model
    torch.cuda.empty_cache()

    # Step 5: Post-processing
    update_progress("post_processing")

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if info.language in punct_model_langs:
        punct_model = PunctuationModel(model="kredor/punctuate-all")
        words_list = list(map(lambda x: x["word"], wsm))
        labeled_words = punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labeled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word
    else:
        logging.warning(
            f"Punctuation restoration not available for {info.language}. "
            "Using original punctuation."
        )

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    # Step 6: Generate outputs in Deepgram-compatible format
    update_progress("generating_output")

    # Build words array (Deepgram format)
    words = []
    for word_dict in wsm:
        speaker_num = word_dict["speaker"]
        words.append({
            "word": word_dict["word"].strip(),
            "start": word_dict["start_time"] / 1000.0,  # Convert ms to seconds
            "end": word_dict["end_time"] / 1000.0,
            "confidence": 0.95,  # We don't have per-word confidence, use high default
            "speaker": speaker_num,
            "speaker_confidence": 0.85,  # Default confidence
            "punctuated_word": word_dict["word"].strip(),
        })

    # Build utterances array (Deepgram format) from sentence mappings
    utterances = []
    for sentence_dict in ssm:
        speaker_str = sentence_dict["speaker"]  # "Speaker 0", "Speaker 1", etc.
        speaker_num = int(speaker_str.split()[-1]) if speaker_str.startswith("Speaker") else 0

        # Get words for this utterance
        utterance_words = [
            w for w in words
            if w["start"] >= sentence_dict["start_time"] / 1000.0
            and w["end"] <= sentence_dict["end_time"] / 1000.0 + 0.1
        ]

        utterances.append({
            "start": sentence_dict["start_time"] / 1000.0,
            "end": sentence_dict["end_time"] / 1000.0,
            "confidence": 0.95,
            "channel": 0,
            "transcript": sentence_dict["text"].strip(),
            "words": utterance_words,
            "speaker": speaker_num,
            "id": str(uuid4()),
        })

    # Build full transcript
    full_transcript = " ".join(w["word"] for w in words)

    # Cleanup temp files
    if os.path.exists(temp_outputs_dir):
        cleanup(temp_outputs_dir)

    update_progress("completed")

    # Return Deepgram-compatible response
    return {
        "metadata": {
            "request_id": str(uuid4()),
            "model_info": {
                "name": model_name,
            },
            "duration": words[-1]["end"] if words else 0,
        },
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": full_transcript,
                            "confidence": 0.95,
                            "words": words,
                        }
                    ]
                }
            ],
            "utterances": utterances,
        },
    }
