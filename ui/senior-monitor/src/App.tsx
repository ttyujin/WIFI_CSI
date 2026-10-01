import { useEffect, useState } from 'react';
import { ProfileSetup } from './components/ProfileSetup';
import { Header } from './components/Header';
import { CurrentStatusCard } from './components/CurrentStatusCard';
import { DailySummary } from './components/DailySummary';
import { ActivityHistory } from './components/ActivityHistory';
import { useActivity } from './hooks/useActivity';
import { loadProfile, saveProfile } from './utils/profile';
import { getProfile, postProfile } from './api/profileApi';
import { dailyTotals } from './utils/time';
import type { SeniorProfile } from './types';

export default function App() {
  const [cached] = useState(loadProfile);
  const [profile, setProfile] = useState<SeniorProfile | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [notice, setNotice] = useState('');
  const [emailEnabled, setEmailEnabled] = useState(false);
  const [editing, setEditing] = useState(false);
  const activity = useActivity(profile !== null);
  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 4000);
    void getProfile(controller.signal).then((result) => {
      if (!active || controller.signal.aborted) return;
      setProfile(result.profile);
      setEmailEnabled(result.email_alerts_enabled);
      if (result.profile) {
        try { saveProfile(result.profile); }
        catch { setNotice('서버 프로필은 불러왔지만 브라우저 캐시에 저장하지 못했어요.'); }
      }
    }).catch(() => {
      if (!active) return;
      setNotice('서버 프로필을 불러오지 못했어요. 저장 시 다시 연결합니다.');
      if (cached?.guardian_email) setProfile({ ...cached, guardian_email: cached.guardian_email });
    }).finally(() => { clearTimeout(timer); if (active) setLoaded(true); });
    return () => { active = false; clearTimeout(timer); controller.abort(); };
  }, [cached]);
  useEffect(() => {
    document.title = profile ? profile.name + ' 어르신 생활 상태' : '어르신 프로필 설정';
  }, [profile]);

  async function onSave(value: SeniorProfile) {
    const result = await postProfile(value);
    if (!result.profile) throw new Error('Missing profile');
    setNotice('');
    try { saveProfile(result.profile); }
    catch { setNotice('서버에는 저장했지만 브라우저 캐시에 저장하지 못했어요. 다음 접속 시 서버에서 다시 불러옵니다.'); }
    setProfile(result.profile);
    setEmailEnabled(result.email_alerts_enabled);
    setEditing(false);
  }
  if (!loaded) return <main className="app"><p role="status">프로필 확인 중…</p></main>;
  if (!profile || editing) return <main className="app">
    {notice && <p role="status" className="history-notice">{notice}</p>}
    {editing && !emailEnabled && <p className="history-notice">이메일 알림 설정 필요</p>}
    <ProfileSetup initial={profile ?? cached} editing={editing} onSave={onSave} onCancel={() => setEditing(false)} />
  </main>;
  const totals = dailyTotals(activity.history, activity.current, activity.duration, activity.now);
  return <main className="app">
    <Header profile={profile} connection={activity.connection} onEdit={() => setEditing(true)} />
    {notice && <p role="status" className="history-notice">{notice}</p>}
    <CurrentStatusCard current={activity.current} profile={profile} duration={activity.duration} connection={activity.connection} />
    <DailySummary totals={totals} loaded={activity.historyLoaded}
      incomplete={activity.incomplete} />
    <ActivityHistory current={activity.current} history={activity.history} duration={activity.duration}
      now={activity.now} error={activity.apiReachable === false} />
  </main>;
}
