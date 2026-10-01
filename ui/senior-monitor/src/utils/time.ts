import type { ActivityHistoryItem, DailyTotals, KnownActivity } from '../types';

const pad = (n: number) => String(n).padStart(2, '0');
export function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  return [Math.floor(total / 3600), Math.floor(total / 60) % 60, total % 60].map(pad).join(':');
}
export function shortDuration(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  return (hours ? hours + '시간 ' : '') + Math.floor(total / 60) % 60 + '분 ' + pad(total % 60) + '초';
}
export function localTime(iso: string, now: number): string {
  const date = new Date(iso);
  const prefix = date.toDateString() === new Date(now).toDateString() ? '' :
    (date.getMonth() + 1) + '/' + date.getDate() + ' ';
  return prefix + pad(date.getHours()) + ':' + pad(date.getMinutes());
}
export function localDayBounds(now: number): [number, number] {
  const start = new Date(now);
  start.setHours(0, 0, 0, 0);
  const end = new Date(start);
  end.setDate(end.getDate() + 1); // Calendar days, including 23/25-hour DST days.
  return [+start, +end];
}
export function sameSegment(a: ActivityHistoryItem, b: KnownActivity): boolean {
  return a.state === b.state && Date.parse(a.started_at) === Date.parse(b.started_at);
}
export function dedupeHistory(history: ActivityHistoryItem[]): ActivityHistoryItem[] {
  const unique = new Map<string, ActivityHistoryItem>();
  for (const row of history) {
    const key = row.state + ':' + Date.parse(row.started_at);
    const previous = unique.get(key);
    if (!previous || Date.parse(row.ended_at) > Date.parse(previous.ended_at)) unique.set(key, row);
  }
  return [...unique.values()].sort((a, b) => Date.parse(b.ended_at) - Date.parse(a.ended_at));
}
export function todayHistory(history: ActivityHistoryItem[], now: number): ActivityHistoryItem[] {
  const [dayStart, dayEnd] = localDayBounds(now);
  return dedupeHistory(history).filter((row) =>
    Date.parse(row.ended_at) > dayStart && Date.parse(row.started_at) < Math.min(dayEnd, now));
}

/**
 * Totals cover only API-provided segments, never a persisted/inferred full day.
 * Timestamp ranges are authoritative for completed records; displayed current
 * seconds are interpolated only up to the server's last confirmed prediction.
 * Do not slice history to the UI's 15-row display limit here.
 */
export function dailyTotals(history: ActivityHistoryItem[], current: KnownActivity | null,
  currentSeconds: number, now: number): DailyTotals {
  const [dayStart, dayEnd] = localDayBounds(now);
  const records = dedupeHistory(history);
  const ranges: Record<'MOVING' | 'STAYING', [number, number][]> = { MOVING: [], STAYING: [] };
  const add = (state: 'MOVING' | 'STAYING', start: number, end: number) => {
    const from = Math.max(dayStart, start);
    const to = Math.min(dayEnd, now, end);
    if (to > from) ranges[state].push([from, to]);
  };
  for (const row of records) add(row.state, Date.parse(row.started_at), Date.parse(row.ended_at));
  // Defensive deduplication: a closed segment wins over overlapping current.
  if (current && !records.some((row) => sameSegment(row, current))) {
    const start = Date.parse(current.started_at);
    add(current.state, start, start + Math.max(0, currentSeconds) * 1000);
  }
  const sum = (intervals: [number, number][]) => {
    intervals.sort((a, b) => a[0] - b[0]);
    let end = -Infinity, milliseconds = 0;
    for (const [from, to] of intervals) {
      milliseconds += Math.max(0, to - Math.max(from, end));
      end = Math.max(end, to);
    }
    return milliseconds / 1000;
  };
  return { MOVING: sum(ranges.MOVING), STAYING: sum(ranges.STAYING) };
}
