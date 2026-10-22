"""Incremental final-channel rendering for Harmony and its older token aliases.

The checkpoint template spells start/channel/message/end/return/call; newer
Harmony tokenizers spell im_start/meta_sep/im_sep/im_end/fim_suffix/ghissue.
Protocol headers and nonfinal channels never become visible answer text.
"""

import re

_MARKERS = re.compile(r"<\|[^<>|]+\|>")
_START = {"<|start|>", "<|im_start|>"}
_MESSAGE = {"<|message|>", "<|im_sep|>"}
_CHANNEL = {"<|channel|>", "<|meta_sep|>"}
_FORMAT = {"<|meta_start|>", "<|format|>"}
_END = {
    "<|end|>",
    "<|im_end|>",
    "<|return|>",
    "<|fim_suffix|>",
    "<|call|>",
    "<|ghissue|>",
    "<|endoftext|>",
}


class HarmonyFinalParser:
    def __init__(self, prompt=""):
        self.buffer = ""
        self.header = "assistant"
        self.in_header = True
        self.visible = False
        # A caller may have explicitly prefilled an assistant final header.
        starts = list(re.finditer(r"<\|(?:start|im_start)\|>assistant", prompt))
        if starts:
            tail = prompt[starts[-1].end() :]
            self.feed(tail)

    def feed(self, piece):
        self.buffer += piece
        result = []
        while self.buffer:
            marker = _MARKERS.search(self.buffer)
            if marker is None:
                # Hold a possible partial control token across tokenizer chunks.
                partial = self.buffer.rfind("<")
                keep = (
                    partial
                    if partial >= 0 and "|>" not in self.buffer[partial:]
                    else len(self.buffer)
                )
                ready, self.buffer = self.buffer[:keep], self.buffer[keep:]
                if self.in_header:
                    self.header += ready
                elif self.visible:
                    result.append(ready)
                break
            ready = self.buffer[: marker.start()]
            token = marker.group()
            self.buffer = self.buffer[marker.end() :]
            if self.in_header:
                self.header += ready
            elif self.visible:
                result.append(ready)
            if token in _START:
                self.header, self.in_header, self.visible = "", True, False
            elif token in _CHANNEL:
                self.header += " channel="
            elif token in _FORMAT:
                self.header += " format="
            elif token in _MESSAGE:
                header = self.header.strip()
                channel = re.search(r"channel=\s*(\w+)", header)
                self.visible = (
                    header.split(" ", 1)[0] == "assistant"
                    and "to=" not in header
                    and (channel is None or channel.group(1) == "final")
                )
                self.in_header = False
            elif token in _END:
                self.in_header, self.visible, self.header = True, False, ""
        return "".join(result)

    def flush(self):
        # An incomplete protocol marker or header must never become answer text.
        text = self.buffer if self.visible and not self.buffer.startswith("<") else ""
        self.buffer = ""
        return text


def final_text(tokenizer, token_ids, *, prompt, final=False):
    parser = HarmonyFinalParser(prompt)
    result = parser.feed(tokenizer.decode(token_ids, skip_special_tokens=False))
    return result + (parser.flush() if final else "")
