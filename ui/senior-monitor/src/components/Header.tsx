import type { ConnectionStatus, SeniorProfile } from '../types';

export function Header({ profile, connection, onEdit }: {
  profile: SeniorProfile; connection: ConnectionStatus; onEdit: () => void;
}) {
  const labels = { LIVE: '실시간 연결됨', RECONNECTING: '다시 연결 중', CONNECTING: '데이터 기다리는 중' };
  return <header className="header">
    <div className="header-top"><p className="eyebrow">일상을 살펴보는 작은 창</p>
      <button className="edit-button" type="button" onClick={onEdit}>프로필 수정</button></div>
    <h1 aria-label={profile.name + ' 어르신 생활 상태'}><span className="senior-name">{profile.name} 어르신 </span>생활 상태<span className="heading-dot" aria-hidden="true">.</span></h1>
    <p className="subtitle">실시간 생활 상태</p>
    <div className={'connection-badge ' + connection} role="status">{labels[connection]}</div>
  </header>;
}
