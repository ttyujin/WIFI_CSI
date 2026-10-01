import { StrictMode } from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import App from './App';
import { PROFILE_KEY } from './utils/profile';
import { IMAGES } from './utils/characters';

const iso = (ms: number) => new Date(ms).toISOString();
function sample(state = 'STAYING', seconds = 120) {
  return { state, connection_status: 'LIVE', is_live: true, last_prediction_at: iso(Date.now()),
    confirmed_duration_seconds: seconds + 2, started_at: iso(Date.now() - (seconds + 2) * 1000),
    updated_at: iso(Date.now()), duration_seconds: seconds, moving_probability: state === 'MOVING' ? 0.97 : 0.63 };
}
function inactive(connection_status: 'CONNECTING' | 'RECONNECTING') {
  return { state: 'UNKNOWN', connection_status, is_live: false, started_at: null,
    updated_at: null, last_prediction_at: null, moving_probability: null,
    duration_seconds: 0, confirmed_duration_seconds: 0 };
}
let backendProfile: { name: string; gender: string; guardian_email: string } | null;
const completeProfile = { name: '홍길동', gender: 'female', guardian_email: 'guardian@example.com' };
let data: { current: unknown; history: unknown };
let fetchMock: ReturnType<typeof vi.fn>;
const save = () => {
  backendProfile = { ...completeProfile };
  localStorage.setItem(PROFILE_KEY, JSON.stringify(backendProfile));
};
const settle = async () => { await act(async () => { await Promise.resolve(); }); };
const advance = async (ms: number) => { await act(async () => { await vi.advanceTimersByTimeAsync(ms); }); };
beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date', 'performance', 'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
  vi.setSystemTime(new Date(2026, 8, 10, 12));
  backendProfile = null;
  data = { current: inactive('CONNECTING'), history: [] };
  fetchMock = vi.fn(async (url: string, options?: RequestInit) => {
    if (url.endsWith('/api/profile')) {
      if (options?.method === 'POST') backendProfile = JSON.parse(String(options.body));
      return { ok: true, json: async () => ({ success: true, profile: backendProfile, email_alerts_enabled: false }) };
    }
    expect(url).toContain('/current?include_history=1');
    if (data.current instanceof Error) throw data.current;
    if (data.history instanceof Error) throw data.history;
    return { ok: true, json: async () => ({ ...(data.current as object), history: data.history }) };
  });
  vi.stubGlobal('fetch', fetchMock);
});

it('first visit requires a profile, previews selected GIF, saves name/type and enters main', async () => {
  render(<App />); await settle();
  expect(screen.getByRole('heading', { name: '어르신 성함을 알려주세요' })).toBeTruthy();
  expect(fetchMock).toHaveBeenCalledTimes(1);
  fireEvent.change(screen.getByLabelText('어르신 이름'), { target: { value: '  홍길동  ' } });
  fireEvent.click(screen.getByRole('button', { name: '다음' }));
  expect(screen.getByAltText('할아버지 캐릭터 미리보기').getAttribute('src')).toBe(IMAGES.grandpa.STAYING);
  fireEvent.click(screen.getByRole('button', { name: /남성/ }));
  fireEvent.change(screen.getByLabelText('보호자 이메일'), { target: { value: 'guardian@example.com' } });
  fireEvent.click(screen.getByRole('button', { name: '시작하기' }));
  await settle();
  expect(JSON.parse(localStorage.getItem(PROFILE_KEY)!)).toEqual({ name: '홍길동', gender: 'male', guardian_email: 'guardian@example.com' });
  expect(screen.getByRole('heading', { name: '홍길동 어르신 생활 상태' })).toBeTruthy();
});

it('revisit skips setup; editing profile keeps backend state, history and polling', async () => {
  save(); data.current = sample('MOVING');
  data.history = [{ state: 'STAYING', started_at: iso(Date.now() - 300000),
    ended_at: iso(Date.now() - 120000), duration_seconds: 180 }];
  const view = render(<App />); await settle();
  expect(screen.queryByRole('button', { name: '시작하기' })).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: '프로필 수정' }));
  fireEvent.change(screen.getByLabelText('어르신 이름'), { target: { value: '김영희' } });
  fireEvent.click(screen.getByRole('button', { name: '다음' }));
  fireEvent.click(screen.getByRole('button', { name: /남성/ }));
  fireEvent.click(screen.getByRole('button', { name: '저장하기' })); await settle();
  expect(screen.getByRole('heading', { name: '김영희 어르신 생활 상태' })).toBeTruthy();
  expect(screen.getByAltText('할아버지 캐릭터 · 활동 중').getAttribute('src')).toBe(IMAGES.grandpa.MOVING);
  expect(screen.getByTestId('staying-total').textContent).toBe('00:03:00');
  expect(fetchMock.mock.calls.filter(([url]) => url.includes('/current?'))).toHaveLength(1);
  view.unmount(); render(<App />); await settle();
  expect(screen.getByRole('heading', { name: '김영희 어르신 생활 상태' })).toBeTruthy();
});

it('uses both states and all four GIFs through profile selection and API transitions', async () => {
  save(); data.current = sample('MOVING');
  render(<App />); await settle();
  expect(screen.getByAltText('할머니 캐릭터 · 활동 중').getAttribute('src')).toBe(IMAGES.grandma.MOVING);
  data.current = sample('STAYING', 3); await advance(1000);
  expect(screen.getByAltText('할머니 캐릭터 · 머무르는 중').getAttribute('src')).toBe(IMAGES.grandma.STAYING);
  fireEvent.click(screen.getByRole('button', { name: '프로필 수정' }));
  fireEvent.click(screen.getByRole('button', { name: '다음' }));
  fireEvent.click(screen.getByRole('button', { name: /남성/ }));
  fireEvent.click(screen.getByRole('button', { name: '저장하기' })); await settle();
  expect(screen.getByAltText('할아버지 캐릭터 · 머무르는 중').getAttribute('src')).toBe(IMAGES.grandpa.STAYING);
  data.current = sample('MOVING', 2); await advance(1000);
  expect(screen.getByAltText('할아버지 캐릭터 · 활동 중').getAttribute('src')).toBe(IMAGES.grandpa.MOVING);
});

it('UNKNOWN has no activity GIF, duration or invented current segment', async () => {
  save(); render(<App />); await settle();
  expect(screen.getByRole('heading', { name: '데이터 수신 대기 중' })).toBeTruthy();
  expect(screen.queryByRole('img')).toBeNull();
  expect(screen.getByTestId('current-duration').textContent).toBe('--:--:--');
  expect(screen.getByTestId('moving-total').textContent).toBe('00:00:00');
  expect(screen.getByText('데이터 기다리는 중')).toBeTruthy();
});

it.each(['MOVING', 'STAYING'])('interpolates %s seconds only up to the last confirmed prediction', async (state) => {
  save(); data.current = sample(state, 180);
  render(<App />); await settle();
  // Hold the next HTTP call pending to exercise display interpolation itself.
  fetchMock.mockImplementation(() => new Promise(() => {}));
  await advance(2000);
  expect(screen.getByTestId('current-duration').textContent).toBe('00:03:02');
  expect(screen.getByTestId(state.toLowerCase() + '-total').textContent).toBe('00:03:02');
  await advance(2000);
  expect(screen.getByTestId('current-duration').textContent).toBe('00:03:02');
});

it('hides invalid current on API errors, preserves completed history and waits for explicit LIVE', async () => {
  save(); data.current = sample('MOVING', 180);
  data.history = [{ state: 'STAYING', started_at: iso(Date.now() - 300000),
    ended_at: iso(Date.now() - 180000), duration_seconds: 120 }];
  render(<App />); await settle();
  data.current = new Error('offline'); data.history = new Error('offline');
  await advance(3000);
  expect(screen.getByRole('heading', { name: '다시 연결 중' })).toBeTruthy();
  expect(screen.queryByRole('img')).toBeNull();
  expect(screen.getByTestId('staying-total').textContent).toBe('00:02:00');
  const frozen = screen.getByTestId('current-duration').textContent;
  await advance(2000);
  expect(screen.getByTestId('current-duration').textContent).toBe(frozen);
  data.current = inactive('RECONNECTING');
  await advance(1000);
  expect(screen.queryByText('실시간 연결됨')).toBeNull();
  expect(screen.queryByRole('img')).toBeNull();
  data.current = sample('STAYING', 1); data.history = [];
  await advance(3000);
  expect(screen.getByText('실시간 연결됨')).toBeTruthy();
  expect(screen.getByRole('heading', { name: '머무르는 중' })).toBeTruthy();
});

it.each(['STAYING', 'MOVING'])('CSI disconnect freezes %s totals and requires backend LIVE on recovery', async (state) => {
  save(); const live = sample(state, 120); data.current = live;
  render(<App />); await settle();
  const completed = { state, started_at: live.started_at, ended_at: live.last_prediction_at, duration_seconds: 122 };
  data.current = inactive('RECONNECTING'); data.history = [completed];
  await advance(1000);
  expect(screen.getByRole('heading', { name: '다시 연결 중' })).toBeTruthy();
  expect(screen.queryByRole('img')).toBeNull();
  expect(document.querySelector('.history-item.current')).toBeNull();
  const total = screen.getByTestId(state.toLowerCase() + '-total');
  expect(total.textContent).toBe('00:02:02');
  await advance(20000);
  expect(total.textContent).toBe('00:02:02');
  expect(screen.getByTestId('current-duration').textContent).toBe('--:--:--');
  data.current = sample(state, 0); await advance(1000);
  expect(screen.getByText('실시간 연결됨')).toBeTruthy();
  expect(screen.getByTestId('current-duration').textContent).toBe('00:00:00');
});

it('StrictMode cleanup prevents duplicate loops and clears every timer on unmount', async () => {
  save(); const view = render(<StrictMode><App /></StrictMode>); await settle();
  await advance(2000);
  expect(fetchMock.mock.calls.filter(([url]) => url.includes('/current?'))).toHaveLength(3);
  expect(fetchMock.mock.calls.filter(([url]) => url.endsWith('/api/profile'))).toHaveLength(2);
  view.unmount();
  expect(vi.getTimerCount()).toBe(0);
  const count = fetchMock.mock.calls.length;
  await advance(10000);
  expect(fetchMock).toHaveBeenCalledTimes(count);
});

it('corrupt cache falls back to wizard; cache failure cannot undo a successful backend save', async () => {
  localStorage.setItem(PROFILE_KEY, '{broken');
  render(<App />); await settle();
  fireEvent.change(screen.getByLabelText('어르신 이름'), { target: { value: '홍길동' } });
  fireEvent.click(screen.getByRole('button', { name: '다음' }));
  fireEvent.click(screen.getByRole('button', { name: /여성/ }));
  fireEvent.change(screen.getByLabelText('보호자 이메일'), { target: { value: 'guardian@example.com' } });
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
  fireEvent.click(screen.getByRole('button', { name: '시작하기' })); await settle();
  expect(screen.getByRole('heading', { name: '홍길동 어르신 생활 상태' })).toBeTruthy();
  expect(screen.getByText(/서버에는 저장했지만/)).toBeTruthy();
  expect(backendProfile?.guardian_email).toBe('guardian@example.com');
});

it('a delayed response from the cleaned-up StrictMode effect cannot overwrite newer data', async () => {
  save();
  let resolveOld!: (value: unknown) => void;
  const old = new Promise((resolve) => { resolveOld = resolve; });
  fetchMock.mockImplementationOnce(() => old);
  data.current = sample('MOVING');
  render(<StrictMode><App /></StrictMode>); await settle();
  expect(screen.getByRole('heading', { name: '활동 중' })).toBeTruthy();
  await act(async () => {
    resolveOld({ ok: true, json: async () => ({ ...sample('STAYING'), history: [] }) });
  });
  expect(screen.getByRole('heading', { name: '활동 중' })).toBeTruthy();
});

it('a timed-out HTTP response cannot revive a live GIF, timer or current row', async () => {
  save(); data.current = sample('MOVING');
  render(<App />); await settle();
  let resolveLate!: (value: unknown) => void;
  fetchMock.mockImplementationOnce(() => new Promise((resolve) => { resolveLate = resolve; }));
  await advance(5000);
  expect(screen.getByRole('heading', { name: '다시 연결 중' })).toBeTruthy();
  expect(screen.queryByRole('img')).toBeNull();
  await act(async () => {
    resolveLate({ ok: true, json: async () => ({ ...sample('STAYING'), history: [] }) });
  });
  expect(screen.queryByText('실시간 연결됨')).toBeNull();
  data.current = inactive('RECONNECTING');
  await advance(1000);
  expect(screen.getByRole('heading', { name: '다시 연결 중' })).toBeTruthy();
  data.current = sample('MOVING', 0);
  await advance(1000);
  expect(screen.getByText('실시간 연결됨')).toBeTruthy();
});

it('legacy cache migrates email only; failed backend POST cannot enter main, retry succeeds once', async () => {
  localStorage.setItem(PROFILE_KEY, JSON.stringify({ name: '기존사용자', characterType: 'grandpa' }));
  render(<StrictMode><App /></StrictMode>); await settle();
  expect(screen.getByRole('heading', { name: '보호자 이메일을 입력해주세요' })).toBeTruthy();
  expect(fetchMock.mock.calls.some(([, options]) => options?.method === 'POST')).toBe(false);
  fireEvent.change(screen.getByLabelText('보호자 이메일'), { target: { value: 'new@example.com' } });
  fetchMock.mockRejectedValueOnce(new Error('offline'));
  fireEvent.click(screen.getByRole('button', { name: '시작하기' })); await settle();
  expect(screen.getByRole('alert').textContent).toContain('저장하지 못했습니다');
  expect(screen.queryByRole('button', { name: '프로필 수정' })).toBeNull();
  expect(backendProfile).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: '시작하기' })); await settle();
  expect(backendProfile).toEqual({ name: '기존사용자', gender: 'male', guardian_email: 'new@example.com' });
  expect(screen.getByRole('heading', { name: '기존사용자 어르신 생활 상태' })).toBeTruthy();
  expect(JSON.parse(localStorage.getItem(PROFILE_KEY)!)).toEqual(backendProfile);
});

it('server profile wins over stale browser cache without automatically writing to the backend', async () => {
  save();
  backendProfile = { name: '서버이름', gender: 'male', guardian_email: 'server@example.com' };
  render(<App />); await settle();
  expect(screen.getByRole('heading', { name: '서버이름 어르신 생활 상태' })).toBeTruthy();
  expect(JSON.parse(localStorage.getItem(PROFILE_KEY)!)).toEqual(backendProfile);
  expect(fetchMock.mock.calls.some(([, options]) => options?.method === 'POST')).toBe(false);
});
