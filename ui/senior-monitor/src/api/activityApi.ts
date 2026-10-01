import type { ActivityHistoryItem, ActivitySnapshot, CurrentActivity, KnownActivityState } from '../types';

export const API_BASE = `http://${window.location.hostname}:8010`;
const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
const knownState = (value: unknown): value is KnownActivityState =>
  value === 'MOVING' || value === 'STAYING';
const date = (value: unknown): value is string =>
  typeof value === 'string' && Number.isFinite(Date.parse(value));
const seconds = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0;

export function parseCurrent(value: unknown): CurrentActivity {
  if (!record(value)) throw new Error('Invalid current response');
  if (value.state === 'UNKNOWN') {
    if (value.is_live !== false || !['CONNECTING', 'RECONNECTING'].includes(String(value.connection_status)) ||
        value.started_at !== null || value.moving_probability !== null || value.duration_seconds !== 0 ||
        value.confirmed_duration_seconds !== 0 ||
        (value.updated_at !== null && !date(value.updated_at)) ||
        (value.last_prediction_at !== null && !date(value.last_prediction_at))) throw new Error('Invalid inactive state');
    return { state: 'UNKNOWN', started_at: null, updated_at: value.updated_at,
      connection_status: value.connection_status === 'CONNECTING' ? 'CONNECTING' : 'RECONNECTING',
      is_live: false, last_prediction_at: value.last_prediction_at,
      moving_probability: null, duration_seconds: 0, confirmed_duration_seconds: 0 };
  }
  if (!knownState(value.state) || !date(value.started_at) || !date(value.updated_at) ||
      !seconds(value.duration_seconds) || Date.parse(value.updated_at) < Date.parse(value.started_at) ||
      value.is_live !== true || value.connection_status !== 'LIVE' || !date(value.last_prediction_at) ||
      Date.parse(value.last_prediction_at) < Date.parse(value.started_at) ||
      !seconds(value.confirmed_duration_seconds) || value.duration_seconds > value.confirmed_duration_seconds ||
      value.confirmed_duration_seconds > (Date.parse(value.last_prediction_at) - Date.parse(value.started_at)) / 1000 + 0.002) {
    throw new Error('Invalid current fields');
  }
  const probability = value.moving_probability;
  if (probability !== null &&
      (typeof probability !== 'number' || !Number.isFinite(probability) || probability < 0 || probability > 1)) {
    throw new Error('Invalid probability');
  }
  // State is authoritative. Never apply a frontend probability threshold.
  return { state: value.state, started_at: value.started_at, updated_at: value.updated_at,
    connection_status: 'LIVE', is_live: true, last_prediction_at: value.last_prediction_at,
    confirmed_duration_seconds: value.confirmed_duration_seconds,
    duration_seconds: value.duration_seconds, moving_probability: probability };
}

export function parseHistory(value: unknown): ActivityHistoryItem[] {
  if (!Array.isArray(value)) throw new Error('Invalid history response');
  return value.map((row: unknown) => {
    if (!record(row) || !knownState(row.state) || !date(row.started_at) ||
        !date(row.ended_at) || !seconds(row.duration_seconds) ||
        Date.parse(row.ended_at) < Date.parse(row.started_at)) throw new Error('Invalid history fields');
    return { state: row.state, started_at: row.started_at, ended_at: row.ended_at,
      duration_seconds: row.duration_seconds };
  }).sort((a, b) => Date.parse(b.ended_at) - Date.parse(a.ended_at));
}

async function get(path: string, signal: AbortSignal): Promise<unknown> {
  const response = await fetch(API_BASE + path, { signal, cache: 'no-store' });
  if (!response.ok) throw new Error('HTTP ' + response.status);
  return response.json();
}
export async function getSnapshot(signal: AbortSignal): Promise<ActivitySnapshot> {
  // One locked backend snapshot eliminates current/history response-order races.
  const value = await get('/api/activity/current?include_history=1', signal);
  if (!record(value)) throw new Error('Invalid snapshot');
  return { current: parseCurrent(value), history: parseHistory(value.history) };
}
