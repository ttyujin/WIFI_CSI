import { act, fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { ProfileSetup } from './ProfileSetup';
import { loadProfile, PROFILE_KEY, parseProfile, validEmail } from '../utils/profile';
import { postProfile, getProfile } from '../api/profileApi';
import { API_BASE } from '../api/activityApi';
import config from '../../vite.config';

const profile = { name: '홍길동', gender: 'female' as const, characterType: 'grandma' as const, guardian_email: 'guardian@example.com' };
function emailStep(save = vi.fn(async () => {})) {
  render(<ProfileSetup initial={null} onSave={save} onCancel={() => {}} />);
  fireEvent.change(screen.getByLabelText('어르신 이름'), { target: { value: ' 홍길동 ' } });
  fireEvent.click(screen.getByRole('button', { name: '다음' }));
  fireEvent.click(screen.getByRole('button', { name: /여성/ }));
  return save;
}

it('step 1 rejects blank/control names and does not expose email yet', () => {
  render(<ProfileSetup initial={null} onSave={vi.fn()} onCancel={() => {}} />);
  fireEvent.click(screen.getByRole('button', { name: '다음' }));
  expect(screen.getByRole('alert').textContent).toContain('1~40자');
  expect(screen.queryByLabelText('보호자 이메일')).toBeNull();
});
it('step 2 shows gender cards and step 3 validates email before saving', async () => {
  const save = emailStep();
  expect(screen.getByRole('heading', { name: '보호자 이메일을 입력해주세요' })).toBeTruthy();
  expect(screen.queryByLabelText('어르신 이름')).toBeNull();
  fireEvent.change(screen.getByLabelText('보호자 이메일'), { target: { value: 'bad' } });
  fireEvent.click(screen.getByRole('button', { name: '시작하기' }));
  expect(save).not.toHaveBeenCalled();
  expect(screen.getByRole('alert').textContent).toContain('올바른');
  fireEvent.change(screen.getByLabelText('보호자 이메일'), { target: { value: ' guardian@example.com ' } });
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: '시작하기' })); });
  expect(save).toHaveBeenCalledWith(profile);
});
it('failed save remains at email step and can retry, with duplicate clicks guarded', async () => {
  const save = vi.fn().mockRejectedValueOnce(new Error('offline')).mockResolvedValue(undefined);
  emailStep(save);
  fireEvent.change(screen.getByLabelText('보호자 이메일'), { target: { value: profile.guardian_email } });
  const submitButton = screen.getByRole('button', { name: '시작하기' });
  await act(async () => {
    fireEvent.click(submitButton);
    fireEvent.click(submitButton);
  });
  expect(save).toHaveBeenCalledTimes(1);
  expect(screen.getByRole('alert').textContent).toContain('저장하지 못했습니다');
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: '시작하기' })); });
  expect(save).toHaveBeenCalledTimes(2);
});
it.each([['grandma', 'female'], ['grandpa', 'male']])('migrates %s cache without deleting name, starting at email only', (characterType, gender) => {
  localStorage.setItem(PROFILE_KEY, JSON.stringify({ name: '기존이름', characterType }));
  const initial = loadProfile();
  expect(initial).toMatchObject({ name: '기존이름', gender, characterType });
  render(<ProfileSetup initial={initial} onSave={vi.fn()} onCancel={() => {}} />);
  expect(screen.getByRole('heading', { name: '보호자 이메일을 입력해주세요' })).toBeTruthy();
  expect(screen.queryByLabelText('어르신 이름')).toBeNull();
});
it('editing preserves email and allows changing all three fields', async () => {
  const save = vi.fn(async () => {});
  render(<ProfileSetup initial={profile} editing onSave={save} onCancel={() => {}} />);
  fireEvent.change(screen.getByLabelText('어르신 이름'), { target: { value: '새이름' } });
  fireEvent.click(screen.getByRole('button', { name: '다음' }));
  fireEvent.click(screen.getByRole('button', { name: /남성/ }));
  expect((screen.getByLabelText('보호자 이메일') as HTMLInputElement).value).toBe(profile.guardian_email);
  fireEvent.change(screen.getByLabelText('보호자 이메일'), { target: { value: 'new@example.com' } });
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: '저장하기' })); });
  expect(save).toHaveBeenCalledWith({ name: '새이름', gender: 'male', characterType: 'grandpa', guardian_email: 'new@example.com' });
});
it('profile API posts only whitelist fields and rejects failed or incomplete saves', async () => {
  const fetchMock = vi.fn(async () => ({ ok: true, json: async () => ({ success: true, profile, email_alerts_enabled: true }) }));
  vi.stubGlobal('fetch', fetchMock);
  await postProfile(profile);
  expect(fetchMock).toHaveBeenCalledWith(API_BASE + '/api/profile', expect.objectContaining({
    method: 'POST', body: JSON.stringify({ name: profile.name, gender: profile.gender, guardian_email: profile.guardian_email }),
  }));
  const got = await getProfile(new AbortController().signal);
  expect(got.profile).toEqual(profile);
  fetchMock.mockResolvedValue({ ok: true, json: async () => ({ success: false, profile, email_alerts_enabled: false }) });
  await expect(postProfile(profile)).rejects.toThrow();
  fetchMock.mockResolvedValue({ ok: false, json: async () => ({ success: true, profile, email_alerts_enabled: true }) });
  await expect(postProfile(profile)).rejects.toThrow();
});
it('LAN addresses and ports stay unchanged; email validation rejects header injection', () => {
  expect(API_BASE).toBe('http://' + window.location.hostname + ':8010');
  expect(config).toMatchObject({ server: { host: '0.0.0.0', port: 8090 } });
  for (const email of ['', 'bad', 'a@-bad.com', 'a@b.com\nBcc: other@example.com', 'a'.repeat(255) + '@b.com']) {
    expect(validEmail(email)).toBe(false);
  }
  expect(validEmail('guardian+test@example.com')).toBe(true);
  expect(parseProfile({ name: '기존', characterType: 'grandma', guardian_email: 'bad' })).not.toHaveProperty('guardian_email');
});

it('progress and keyboard focus follow the current step', () => {
  render(<ProfileSetup initial={null} onSave={vi.fn()} onCancel={() => {}} />);
  const currentStep = () => screen.getByRole('list', { name: '프로필 설정 단계' }).querySelector('[aria-current="step"]');
  expect(currentStep()?.textContent).toContain('이름');
  expect(document.activeElement).toBe(screen.getByLabelText('어르신 이름'));
  fireEvent.change(screen.getByLabelText('어르신 이름'), { target: { value: '홍길동' } });
  fireEvent.click(screen.getByRole('button', { name: '다음' }));
  expect(currentStep()?.textContent).toContain('성별');
  expect(document.activeElement).toBe(screen.getByRole('heading'));
  fireEvent.click(screen.getByRole('button', { name: /여성/ }));
  expect(currentStep()?.textContent).toContain('이메일');
  expect(document.activeElement).toBe(screen.getByLabelText('보호자 이메일'));
});

it('previous retains entered values and selected gender; cancel never saves', () => {
  const save = vi.fn();
  const cancel = vi.fn();
  render(<ProfileSetup initial={profile} editing onSave={save} onCancel={cancel} />);
  fireEvent.click(screen.getByRole('button', { name: '다음' }));
  expect(screen.getByRole('button', { name: /여성/ }).getAttribute('aria-pressed')).toBe('true');
  fireEvent.click(screen.getByRole('button', { name: /남성/ }));
  fireEvent.change(screen.getByLabelText('보호자 이메일'), { target: { value: 'changed@example.com' } });
  fireEvent.click(screen.getByRole('button', { name: '이전' }));
  expect(screen.getByRole('button', { name: /남성/ }).getAttribute('aria-pressed')).toBe('true');
  expect(screen.getByRole('button', { name: /여성/ }).getAttribute('aria-pressed')).toBe('false');
  fireEvent.click(screen.getByRole('button', { name: /남성/ }));
  expect((screen.getByLabelText('보호자 이메일') as HTMLInputElement).value).toBe('changed@example.com');
  fireEvent.click(screen.getByRole('button', { name: '이전' }));
  fireEvent.click(screen.getByRole('button', { name: '이전' }));
  expect((screen.getByLabelText('어르신 이름') as HTMLInputElement).value).toBe(profile.name);
  fireEvent.click(screen.getByRole('button', { name: '취소' }));
  expect(cancel).toHaveBeenCalledTimes(1);
  expect(save).not.toHaveBeenCalled();
});

it('invalid fields expose their error and helper through accessible descriptions', () => {
  render(<ProfileSetup initial={null} onSave={vi.fn()} onCancel={() => {}} />);
  fireEvent.click(screen.getByRole('button', { name: '다음' }));
  const input = screen.getByLabelText('어르신 이름');
  expect(input.getAttribute('aria-invalid')).toBe('true');
  expect(input.getAttribute('aria-describedby')?.split(' ')).toEqual(['name-help', 'profile-error']);
  expect(screen.getByRole('alert').id).toBe('profile-error');
});

it('pending save disables navigation, cancel, input and submit without losing its label', async () => {
  let finish!: () => void;
  const save = vi.fn(() => new Promise<void>((resolve) => { finish = resolve; }));
  render(<ProfileSetup initial={profile} editing onSave={save} onCancel={vi.fn()} />);
  fireEvent.click(screen.getByRole('button', { name: '다음' }));
  fireEvent.click(screen.getByRole('button', { name: /여성/ }));
  fireEvent.click(screen.getByRole('button', { name: '저장하기' }));
  for (const name of ['저장 중…', '이전', '취소']) {
    expect((screen.getByRole('button', { name }) as HTMLButtonElement).disabled).toBe(true);
  }
  expect((screen.getByLabelText('보호자 이메일') as HTMLInputElement).disabled).toBe(true);
  expect(screen.getByLabelText('보호자 이메일').closest('form')?.getAttribute('aria-busy')).toBe('true');
  await act(async () => { finish(); });
  expect((screen.getByRole('button', { name: '저장하기' }) as HTMLButtonElement).disabled).toBe(false);
});
