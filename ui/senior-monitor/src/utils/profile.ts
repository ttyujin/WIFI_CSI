import type { ProfileDraft, SeniorProfile } from '../types';

export const PROFILE_KEY = 'senior-monitor.profile.v1';
export function validName(name: string): boolean {
  return name.trim().length > 0 && name.trim().length <= 40 && !/[\u0000-\u001f\u007f]/.test(name);
}
export function validEmail(value: string): boolean {
  if (!value || value.length > 254 || !/^[\x00-\x7f]+$/.test(value)) return false;
  const parts = value.split('@');
  if (parts.length !== 2) return false;
  const [local, domain] = parts;
  return local.length <= 64 && /^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+$/.test(local) &&
    !local.startsWith('.') && !local.endsWith('.') && !local.includes('..') &&
    domain.includes('.') && domain.split('.').every((part) => /^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$/.test(part));
}
export function parseProfile(value: unknown): ProfileDraft | null {
  if (!value || typeof value !== 'object' || !('name' in value) || typeof value.name !== 'string' || !validName(value.name)) return null;
  const gender = 'gender' in value ? value.gender : 'characterType' in value ?
    value.characterType === 'grandma' ? 'female' : value.characterType === 'grandpa' ? 'male' : null : null;
  if (gender !== 'female' && gender !== 'male') return null;
  const email = 'guardian_email' in value && typeof value.guardian_email === 'string' ? value.guardian_email.trim() : '';
  return { name: value.name.trim(), gender, characterType: gender === 'female' ? 'grandma' : 'grandpa',
    ...(validEmail(email) ? { guardian_email: email } : {}) };
}
export function loadProfile(): ProfileDraft | null {
  try { return parseProfile(JSON.parse(localStorage.getItem(PROFILE_KEY) ?? 'null')); }
  catch { return null; }
}
export function saveProfile(profile: SeniorProfile): void {
  // Cache only whitelisted UI data, never credentials or arbitrary API fields.
  localStorage.setItem(PROFILE_KEY, JSON.stringify({
    name: profile.name, gender: profile.gender, guardian_email: profile.guardian_email,
  }));
}
