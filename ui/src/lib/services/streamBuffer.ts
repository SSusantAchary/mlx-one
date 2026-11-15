/** Bound Markdown/DOM updates without dropping text on completion or abort. */
export function streamBuffer(emit: (text: string, thought: string) => void, delay = 80) {
  let text = '', thought = '';
  let timer: ReturnType<typeof setTimeout> | undefined;
  function flush() {
    if (timer !== undefined) clearTimeout(timer);
    timer = undefined;
    const next = text, reasoning = thought;
    text = ''; thought = '';
    if (next || reasoning) emit(next, reasoning);
  }
  return {
    push(piece: string, reasoning: string) {
      text += piece; thought += reasoning;
      if (timer === undefined) timer = setTimeout(flush, delay);
    }, flush
  };
}
