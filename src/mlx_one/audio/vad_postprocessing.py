"""Silero timestamp policy, adapted verbatim from pinned utils_vad.py.

Copyright (c) 2020-present Silero Team. MIT; see models/audio/silero/LICENSE.
Source commit 60b7ffa243625ebdc1070275a29f18c87843786a. Changes: Python
list annotation, plotting explicitly unavailable. Public VAD restricts 16 kHz.
"""


def get_speech_timestamps_from_probs(
    speech_probs: list[float],
    sampling_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
    time_resolution: int = 1,
    visualize_probs: bool = False,
    neg_threshold: float = None,
    min_silence_at_max_speech: int = 98,
    use_max_poss_sil_at_max_speech: bool = True,
    audio_length_samples: int = None,
    step: int = 1,
):
    """
    Convert a sequence of per-frame speech probabilities into speech timestamps.

    Parameters
    ----------
    speech_probs: list[float]
        Per-frame speech probabilities, one per window_size_samples frame,
        in order. window_size_samples is 512 for 16000 Hz and 256 for 8000 Hz.

    audio_length_samples: int (default - None)
        Total number of samples in the original audio (at ``sampling_rate``).
        If None, it is estimated as len(speech_probs) * window_size_samples.

    step: int (default - 1)
        Decimation step applied to the input audio before framing (used when
        the caller down-sampled a multiple-of-16000 rate to 16000). Output
        sample coordinates are multiplied by ``step`` to map back.

    All other parameters match ``get_speech_timestamps``.

    Returns
    ----------
    speeches: list of dicts
        List containing speech beginnings/endings (samples or seconds).
    """
    window_size_samples = 512 if sampling_rate == 16000 else 256
    if audio_length_samples is None:
        audio_length_samples = len(speech_probs) * window_size_samples

    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * min_silence_at_max_speech / 1000

    triggered = False
    speeches = []
    current_speech = {}

    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)
    temp_end = 0  # to save potential segment end (and tolerate some silence)
    prev_end = next_start = (
        0  # to save potential segment limits in case of maximum segment size reached
    )
    possible_ends = []

    for i, speech_prob in enumerate(speech_probs):
        cur_sample = window_size_samples * i

        # If speech returns, record sufficiently long candidate silence and clear temp_end.
        if (speech_prob >= threshold) and temp_end:
            sil_dur = cur_sample - temp_end
            if sil_dur > min_silence_samples_at_max_speech:
                possible_ends.append((temp_end, sil_dur))
            temp_end = 0
            if next_start < prev_end:
                next_start = cur_sample

        # Start of speech
        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech["start"] = cur_sample
            continue

        # Max speech length reached: decide where to cut
        if triggered and (cur_sample - current_speech["start"] > max_speech_samples):
            if use_max_poss_sil_at_max_speech and possible_ends:
                prev_end, dur = max(
                    possible_ends, key=lambda x: x[1]
                )  # use the longest possible silence segment in the current speech chunk
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                next_start = prev_end + dur

                if (
                    next_start < prev_end + cur_sample
                ):  # previously reached silence (< neg_thres) and is still not speech (< thres)
                    current_speech["start"] = next_start
                else:
                    triggered = False
                prev_end = next_start = temp_end = 0
                possible_ends = []
            else:
                # Legacy max-speech cut: prefer last valid silence (prev_end) if available.
                if prev_end:
                    current_speech["end"] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    if next_start < prev_end:
                        triggered = False
                    else:
                        current_speech["start"] = next_start
                    prev_end = next_start = temp_end = 0
                    possible_ends = []
                else:
                    # No prev_end -> fallback to cutting at current sample
                    current_speech["end"] = cur_sample
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    possible_ends = []
                    continue

        # Silence detection while in speech
        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = cur_sample
            sil_dur_now = cur_sample - temp_end

            if (
                not use_max_poss_sil_at_max_speech
                and sil_dur_now > min_silence_samples_at_max_speech
            ):  # condition to avoid cutting in very short silence
                prev_end = temp_end

            if sil_dur_now < min_silence_samples:
                continue
            else:
                current_speech["end"] = temp_end
                if (current_speech["end"] - current_speech["start"]) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                possible_ends = []
                continue

    if current_speech and (audio_length_samples - current_speech["start"]) > min_speech_samples:
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(min(audio_length_samples, speech["end"] + speech_pad_samples))
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(min(audio_length_samples, speech["end"] + speech_pad_samples))

    if return_seconds:
        audio_length_seconds = audio_length_samples / sampling_rate
        for speech_dict in speeches:
            speech_dict["start"] = max(
                round(speech_dict["start"] / sampling_rate, time_resolution), 0
            )
            speech_dict["end"] = min(
                round(speech_dict["end"] / sampling_rate, time_resolution), audio_length_seconds
            )
    elif step > 1:
        for speech_dict in speeches:
            speech_dict["start"] *= step
            speech_dict["end"] *= step

    if visualize_probs:
        raise ValueError("visualization is not part of native VAD inference")

    return speeches
