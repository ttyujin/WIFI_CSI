import type { ConnectionStatus, KnownActivity, SeniorProfile } from '../types';
import { IMAGES, STATE_TEXT } from '../utils/characters';
import { formatDuration } from '../utils/time';
import { CharacterImage } from './CharacterImage';

export function CurrentStatusCard({ current, profile, duration, connection }: {
  current: KnownActivity | null; profile: SeniorProfile; duration: number; connection: ConnectionStatus;
}) {
  const reconnecting = connection === 'RECONNECTING';
  return <section className={'status-card ' + (current?.state.toLowerCase() ?? 'unknown')} aria-labelledby="status-title">
    <div className="card-top"><span className="section-label">현재 상태</span>
      <span className="state-code">{current?.state ?? '대기'}</span></div>
    <div className="character-area">
      {current
        ? <CharacterImage src={IMAGES[profile.characterType][current.state]}
          alt={(profile.characterType === 'grandma' ? '할머니' : '할아버지') + ' 캐릭터 · ' + STATE_TEXT[current.state].name} />
        : <div className="character-placeholder">
          <svg viewBox="0 0 80 80" aria-hidden="true"><path d="M16 31Q40 9 64 31M25 43Q40 29 55 43M34 54Q40 49 46 54"/><circle cx="40" cy="65" r="3"/></svg>
          <span>{reconnecting ? '생활 신호 연결을 확인하고 있어요' : '첫 생활 신호를 기다리고 있어요'}</span>
        </div>}
    </div>
    <div className="status-copy" aria-live="polite" aria-atomic="true">
      <h2 id="status-title">{current ? STATE_TEXT[current.state].name : reconnecting ? '다시 연결 중' : '데이터 수신 대기 중'}</h2>
      <p>{current ? STATE_TEXT[current.state].description : '연결이 없는 시간은 활동 시간에 포함하지 않아요.'}</p>
    </div>
    <div className="duration-box"><p>현재 상태 지속시간</p>
      <strong data-testid="current-duration">{current ? formatDuration(duration) : '--:--:--'}</strong></div>
    <p className="update-text">{current
      ? '마지막 분석 ' + new Date(current.updated_at).toLocaleString('ko-KR')
      : '유효한 CSI 분석 결과를 기다리고 있어요.'}</p>
  </section>;
}
