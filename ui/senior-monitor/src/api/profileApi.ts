import { API_BASE } from './activityApi';
import { parseProfile, validEmail } from '../utils/profile';
import type { SeniorProfile } from '../types';

export interface ProfileResponse { profile: SeniorProfile | null; email_alerts_enabled: boolean }
function parseResponse(value: unknown): ProfileResponse {
  if (!value || typeof value !== 'object' || !('profile' in value) ||
      !('email_alerts_enabled' in value) || typeof value.email_alerts_enabled !== 'boolean') {
    throw new Error('잘못된 프로필 응답입니다.');
  }
  const profile = parseProfile(value.profile);
  if (value.profile !== null && (!profile || !validEmail(profile.guardian_email ?? ''))) {
    throw new Error('잘못된 프로필 응답입니다.');
  }
  return { profile: profile ? { ...profile, guardian_email: profile.guardian_email! } : null,
    email_alerts_enabled: value.email_alerts_enabled };
}

export async function getProfile(signal: AbortSignal): Promise<ProfileResponse> {
  const response = await fetch(API_BASE + '/api/profile', { signal, cache: 'no-store' });
  if (!response.ok) throw new Error('프로필을 불러오지 못했습니다.');
  return parseResponse(await response.json());
}

export async function postProfile(profile: SeniorProfile): Promise<ProfileResponse> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(API_BASE + '/api/profile', {
      method: 'POST', signal: controller.signal, headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: profile.name, gender: profile.gender, guardian_email: profile.guardian_email }),
    });
    if (!response.ok) throw new Error('저장하지 못했습니다. 다시 시도해주세요.');
    const value: unknown = await response.json();
    if (controller.signal.aborted) throw new Error('저장 응답 시간이 초과되었습니다.');
    if (!value || typeof value !== 'object' || !('success' in value) || value.success !== true) {
      throw new Error('저장하지 못했습니다. 다시 시도해주세요.');
    }
    const result = parseResponse(value);
    if (!result.profile) throw new Error('저장된 프로필이 없습니다.');
    return result;
  } finally { clearTimeout(timer); }
}
