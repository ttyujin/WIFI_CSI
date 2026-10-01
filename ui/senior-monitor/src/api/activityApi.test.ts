import { expect, it } from 'vitest';
import { parseCurrent, parseHistory } from './activityApi';
import { parseProfile } from '../utils/profile';

const fixture = { state: 'STAYING', is_live: true, connection_status: 'LIVE',
  last_prediction_at: '2026-09-10T10:01:00+09:00', confirmed_duration_seconds: 60, started_at: '2026-09-10T10:00:00+09:00',
  updated_at: '2026-09-10T10:01:00+09:00', moving_probability: 0.63, duration_seconds: 60 };

it('preserves backend state regardless of probability and keeps UNKNOWN distinct', () => {
  expect(parseCurrent({ ...fixture, moving_probability: 0.99 }).state).toBe('STAYING');
  expect(parseCurrent({ state: 'UNKNOWN', is_live: false, connection_status: 'CONNECTING',
    started_at: null, updated_at: null, last_prediction_at: null, moving_probability: null,
    duration_seconds: 0, confirmed_duration_seconds: 0 }).state).toBe('UNKNOWN');
});
it('rejects malformed or unsupported current/history fields', () => {
  for (const value of [null, {}, { ...fixture, state: 'INVALID' },
    { ...fixture, duration_seconds: null }, { ...fixture, updated_at: 'bad' },
    { ...fixture, moving_probability: NaN }, { ...fixture, is_live: false },
    { ...fixture, connection_status: 'RECONNECTING' }, { ...fixture, confirmed_duration_seconds: 90 }]) expect(() => parseCurrent(value)).toThrow();
  expect(() => parseHistory([{}])).toThrow();
  expect(() => parseHistory({})).toThrow();
});
it('normalizes names and rejects unusable stored profiles', () => {
  expect(parseProfile({ name: '  홍길동  ', characterType: 'grandpa' }))
    .toEqual({ name: '홍길동', gender: 'male', characterType: 'grandpa' });
  for (const p of [null, {}, { name: '', characterType: 'grandma' },
    { name: '홍길동', characterType: 'invalid' }]) expect(parseProfile(p)).toBeNull();
});
