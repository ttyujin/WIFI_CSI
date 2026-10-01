import { describe, expect, it } from 'vitest';
import { dailyTotals, formatDuration, localDayBounds, todayHistory } from './time';
import type { ActivityHistoryItem, KnownActivity, KnownActivityState } from '../types';

// SYNTHETIC UI-only fixtures, never sent to backend/model/recording paths.
const local = (day: number, hour = 0, minute = 0, second = 0) => new Date(2026, 8, day, hour, minute, second).getTime();
const iso = (value: number) => new Date(value).toISOString();
const row = (state: KnownActivityState, start: number, end: number): ActivityHistoryItem => ({
  state, started_at: iso(start), ended_at: iso(end), duration_seconds: (end - start) / 1000,
});
const current = (state: KnownActivityState, start: number): KnownActivity => ({
  state, connection_status: 'LIVE', is_live: true, last_prediction_at: iso(start), confirmed_duration_seconds: 0,
  started_at: iso(start), updated_at: iso(start), duration_seconds: 0, moving_probability: 0.5,
});

describe('local timestamp daily aggregation', () => {
  it('clips yesterday 23:50 through today 00:10 to ten minutes', () => {
    expect(dailyTotals([row('STAYING', local(9, 23, 50), local(10, 0, 10))],
      null, 0, local(10, 12))).toEqual({ STAYING: 600, MOVING: 0 });
  });
  it('includes all completed records and current elapsed seconds, independently by state', () => {
    const records = [row('STAYING', local(10, 8), local(10, 9)),
      row('MOVING', local(10, 9), local(10, 9, 10))];
    const c = current('MOVING', local(10, 10));
    expect(dailyTotals(records, c, 180, local(10, 10, 3))).toEqual({ STAYING: 3600, MOVING: 780 });
    expect(dailyTotals(records, c, 181, local(10, 10, 3, 1)).MOVING).toBe(781);
  });
  it('clips ongoing segment at local midnight and resets at a day boundary', () => {
    const c = current('STAYING', local(9, 23, 50));
    expect(dailyTotals([], c, 900, local(10, 0, 5)).STAYING).toBe(300);
    expect(dailyTotals([], c, 600, local(10)).STAYING).toBe(0);
    expect(dailyTotals([], c, 601, local(10, 0, 0, 1)).STAYING).toBe(1);
  });
  it('has no arbitrary 15-record cap in totals, but returns newest first for the list', () => {
    const records = Array.from({ length: 20 }, (_, i) =>
      row('MOVING', local(10, 1, i), local(10, 1, i + 1)));
    expect(dailyTotals(records, null, 0, local(10, 12)).MOVING).toBe(1200);
    expect(todayHistory(records, local(10, 12))[0]).toEqual(records[19]);
    expect(todayHistory(records, local(10, 12)).slice(0, 15)).toHaveLength(15);
  });
  it('does not double-count closed current or duplicated completed segments across polls', () => {
    const closed = row('STAYING', local(10, 10), local(10, 10, 5));
    expect(dailyTotals([closed, { ...closed }], current('STAYING', local(10, 10)),
      360, local(10, 10, 6))).toEqual({ STAYING: 300, MOVING: 0 });
  });
  it('does not include previous-day, future, or midnight-ended intervals', () => {
    const records = [row('MOVING', local(9, 10), local(9, 11)),
      row('MOVING', local(9, 23), local(10)), row('STAYING', local(11), local(11, 1))];
    expect(dailyTotals(records, null, 0, local(10, 12))).toEqual({ STAYING: 0, MOVING: 0 });
    expect(todayHistory(records, local(10, 12))).toHaveLength(0);
  });
  it('uses calendar boundaries rather than UTC midnight or a fixed 86400 seconds', () => {
    const now = local(10, 12), [start, end] = localDayBounds(now);
    expect(new Date(start).getHours()).toBe(0);
    expect(new Date(end).getDate()).toBe(11);
    expect(start).toBe(local(10));
  });
  it('formats duration with hours that do not wrap at 24', () => {
    expect(formatDuration(763)).toBe('00:12:43');
    expect(formatDuration(90061)).toBe('25:01:01');
  });
  it('excludes a disconnected midnight gap, even when both surrounding states match', () => {
    const records = [row('STAYING', local(9, 23, 55), local(9, 23, 59, 59)),
      row('STAYING', local(10, 0, 5), local(10, 0, 10))];
    expect(dailyTotals(records, null, 0, local(10, 0, 20))).toEqual({ STAYING: 300, MOVING: 0 });
  });
});
