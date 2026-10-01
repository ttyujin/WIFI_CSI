import type { ActivityHistoryItem, KnownActivity } from '../types';
import { localTime, sameSegment, shortDuration, todayHistory } from '../utils/time';
import { STATE_TEXT } from '../utils/characters';

export function ActivityHistory({ current, history, duration, now, error }: {
  current: KnownActivity | null; history: ActivityHistoryItem[]; duration: number;
  now: number; error: boolean;
}) {
  const rows = todayHistory(history, now);
  // Defensive deduplication; current/history arrive in one atomic snapshot.
  const closedCurrent = current && rows.some((row) => sameSegment(row, current));
  return <section className="history-card" aria-labelledby="history-title">
    <div className="history-header"><div><p className="eyebrow">하루의 흐름</p><h2 id="history-title">오늘의 활동 기록</h2></div>
      <span className="today-text">{new Date(now).toLocaleDateString('ko-KR', { month: 'long', day: 'numeric' })}</span></div>
    {error && <p className="history-notice" role="status">활동 기록을 다시 불러오는 중이에요. 마지막 기록을 유지합니다.</p>}
    <ol className="history-list">
      {current && !closedCurrent && <li className={'history-item current ' + current.state.toLowerCase()}>
        <div className="history-time">{localTime(current.started_at, now)} ~ 현재</div>
        <div className="history-details"><span className="history-name">{STATE_TEXT[current.state].name}</span>
          <span className="history-duration">{shortDuration(duration)}</span></div>
      </li>}
      {rows.slice(0, 15).map((row) => <li className={'history-item ' + row.state.toLowerCase()} key={row.state + row.started_at}>
        <div className="history-time">{localTime(row.started_at, now)} ~ {localTime(row.ended_at, now)}</div>
        <div className="history-details"><span className="history-name">{STATE_TEXT[row.state].name}</span>
          <span className="history-duration">{shortDuration(row.duration_seconds)}</span></div>
      </li>)}
    </ol>
    {rows.length === 0 && <p className="history-empty">아직 저장된 활동 기록이 없습니다.</p>}
    <p className="history-footnote">오늘에 걸친 완료 기록을 최근 15개까지 보여드려요.</p>
  </section>;
}
