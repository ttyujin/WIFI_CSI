export type KnownActivityState = 'MOVING' | 'STAYING';
export type ConnectionStatus = 'CONNECTING' | 'LIVE' | 'RECONNECTING';
export type CharacterType = 'grandma' | 'grandpa';

export interface ProfileDraft {
  name: string;
  gender: 'female' | 'male';
  characterType: CharacterType;
  guardian_email?: string;
}

export interface SeniorProfile extends ProfileDraft {
  guardian_email: string;
}

export interface KnownActivity {
  connection_status: 'LIVE';
  is_live: true;
  last_prediction_at: string;
  confirmed_duration_seconds: number;
  state: KnownActivityState;
  started_at: string;
  moving_probability: number | null;
  updated_at: string;
  duration_seconds: number;
}

export interface UnknownActivity {
  connection_status: 'CONNECTING' | 'RECONNECTING';
  is_live: false;
  last_prediction_at: string | null;
  confirmed_duration_seconds: 0;
  state: 'UNKNOWN';
  started_at: null;
  moving_probability: null;
  updated_at: string | null;
  duration_seconds: 0;
}

export type CurrentActivity = KnownActivity | UnknownActivity;

export interface ActivityHistoryItem {
  state: KnownActivityState;
  started_at: string;
  ended_at: string;
  duration_seconds: number;
}

export type DailyTotals = Record<KnownActivityState, number>;
export interface ActivitySnapshot {
  current: CurrentActivity;
  history: ActivityHistoryItem[];
}
