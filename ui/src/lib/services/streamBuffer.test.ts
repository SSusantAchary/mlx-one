import { afterEach, expect, it, vi } from 'vitest';
import { streamBuffer } from './streamBuffer';
afterEach(() => vi.useRealTimers());
it('coalesces chunks in order and flushes the final tail exactly once', () => {
  vi.useFakeTimers(); const emit=vi.fn(); const buffer=streamBuffer(emit);
  buffer.push('a','x');buffer.push('b','y');expect(emit).not.toHaveBeenCalled();
  vi.advanceTimersByTime(80);expect(emit).toHaveBeenNthCalledWith(1,'ab','xy');
  buffer.push('c','z');buffer.flush();buffer.flush();vi.runAllTimers();
  expect(emit).toHaveBeenNthCalledWith(2,'c','z');expect(emit).toHaveBeenCalledTimes(2);
});
it('retains a partial response when flushed on cancellation', () => {
  vi.useFakeTimers();const emit=vi.fn();const buffer=streamBuffer(emit);
  buffer.push('partial','');buffer.flush();vi.runAllTimers();
  expect(emit).toHaveBeenCalledExactlyOnceWith('partial','');
});
