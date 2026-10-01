import { useEffect, useRef, useState, type FormEvent } from 'react';
import type { ProfileDraft, SeniorProfile } from '../types';
import { IMAGES } from '../utils/characters';
import { validEmail, validName } from '../utils/profile';
import { CharacterImage } from './CharacterImage';

interface Props {
  initial: ProfileDraft | null;
  editing?: boolean;
  onSave: (profile: SeniorProfile) => Promise<void>;
  onCancel: () => void;
}
export function ProfileSetup({ initial, editing = false, onSave, onCancel }: Props) {
  const [step, setStep] = useState(initial && !initial.guardian_email && !editing ? 3 : 1);
  const [name, setName] = useState(initial?.name ?? '');
  const [gender, setGender] = useState<'female' | 'male' | null>(initial?.gender ?? null);
  const [email, setEmail] = useState(initial?.guardian_email ?? '');
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const inFlight = useRef(false);
  const heading = useRef<HTMLHeadingElement>(null);
  const nameInput = useRef<HTMLInputElement>(null);
  const emailInput = useRef<HTMLInputElement>(null);
  useEffect(() => {
    // Move keyboard focus with the screen, without changing navigation/save logic.
    (step === 1 ? nameInput.current : step === 3 ? emailInput.current : heading.current)?.focus();
  }, [step]);
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (inFlight.current) return;
    setError('');
    if (!validName(name)) { setError('이름을 1~40자로 입력해주세요.'); return; }
    if (step === 1) { setName(name.trim()); setStep(2); return; }
    if (step === 2 || !gender) { setError('성별을 선택해주세요.'); return; }
    if (!validEmail(email.trim())) { setError('올바른 보호자 이메일을 입력해주세요.'); return; }
    inFlight.current = true; setSaving(true);
    try {
      await onSave({ name: name.trim(), gender, guardian_email: email.trim(),
        characterType: gender === 'female' ? 'grandma' : 'grandpa' });
    } catch {
      setError('저장하지 못했습니다. 다시 시도해주세요.');
    } finally { inFlight.current = false; setSaving(false); }
  }
  const title = step === 1 ? '어르신 성함을 알려주세요' : step === 2 ?
    name.trim() + ' 어르신의 성별을 선택해주세요' : '보호자 이메일을 입력해주세요';
  return <section className="profile-screen" aria-labelledby="profile-title">
    <div className="wizard-brand">
      <span className="wizard-mark" aria-hidden="true">
        <svg viewBox="0 0 32 32"><path d="M7 12c0-5 7-6 9-1 2-5 9-4 9 1 0 5-9 11-9 11S7 17 7 12Z"/><path d="M5 6a20 20 0 0 1 22 0"/></svg>
      </span>
      <span>일상을 잇는 작은 안심</span>
    </div>
    <div className="wizard-progress-heading">
      <p className="eyebrow">어르신 프로필 {editing ? '수정' : '설정'}</p>
      <span className="wizard-step-count">STEP <strong>{step}</strong> / 3</span>
    </div>
    <ol className="wizard-progress" aria-label="프로필 설정 단계">
      {['이름', '성별', '이메일'].map((label, index) => <li key={label}
        className={index + 1 < step ? 'complete' : index + 1 === step ? 'current' : ''}
        aria-current={index + 1 === step ? 'step' : undefined}>
        <span className="wizard-step-dot" aria-hidden="true">{index + 1 < step ? '✓' : index + 1}</span>
        <span>{label}</span>
      </li>)}
    </ol>
    <div key={step} className="wizard-page">
    <div className="wizard-heading">
      <h1 id="profile-title" ref={heading} tabIndex={-1}>{title}</h1>
      <p className="profile-intro">{step === 1 ? '소중한 분의 일상을 함께 살펴볼게요.' : step === 2 ?
        '선택한 캐릭터로 어르신의 하루를 보여드려요.' :
        '생활 상태가 일정 시간 이상 지속되면 이 이메일로 확인 안내를 보내드려요.'}</p>
    </div>
    <form onSubmit={(event) => { void submit(event); }} noValidate aria-busy={saving}>
      <div className={'wizard-panel' + (step === 2 ? ' wizard-panel-selection' : '')}>
      {step === 1 && <div className="wizard-field">
        <label className="field-label" htmlFor="senior-name">어르신 이름</label>
        <input id="senior-name" ref={nameInput} value={name} onChange={(e) => setName(e.target.value)}
          maxLength={40} autoComplete="off" placeholder="예: 홍길동" enterKeyHint="next"
          aria-invalid={Boolean(error && !validName(name))}
          aria-describedby={'name-help' + (error ? ' profile-error' : '')} />
        <p id="name-help" className="wizard-helper">화면에 표시할 성함을 입력해주세요.</p>
      </div>}
      {step === 2 && <fieldset><legend className="field-label">성별</legend>
        <div className="person-selector">
          {(['female', 'male'] as const).map((value) => <button key={value} type="button"
            className="gender-card"
            aria-pressed={gender === value} onClick={() => { setGender(value); setStep(3); setError(''); }}>
            <span className="gender-check" aria-hidden="true">{gender === value ? '✓' : ''}</span>
            <span className="gender-image">
              <CharacterImage src={IMAGES[value === 'female' ? 'grandma' : 'grandpa'].STAYING}
                alt={value === 'female' ? '할머니 캐릭터 미리보기' : '할아버지 캐릭터 미리보기'} />
            </span>
            <span className="gender-label">{value === 'female' ? '여성' : '남성'}</span>
            <span className="gender-hint" aria-hidden="true">{gender === value ? '선택됨' : '선택하기'}</span>
          </button>)}
        </div>
        <p className="wizard-helper">카드를 선택하면 다음 단계로 이동해요.</p>
      </fieldset>}
      {step === 3 && <div className="wizard-field">
        <label className="field-label" htmlFor="guardian-email">보호자 이메일</label>
        <input id="guardian-email" ref={emailInput} type="email" value={email} onChange={(e) => setEmail(e.target.value)}
          maxLength={254} autoComplete="email" placeholder="guardian@example.com" disabled={saving}
          inputMode="email" enterKeyHint="done" autoCapitalize="none" spellCheck={false}
          aria-invalid={Boolean(error && !validEmail(email.trim()))}
          aria-describedby={'email-help' + (error ? ' profile-error' : '')} />
        <p id="email-help" className="wizard-helper">안내를 받을 수 있는 이메일인지 확인해주세요.</p>
      </div>}
      {error && <p id="profile-error" className="form-error" role="alert">{error}</p>}
      </div>
      <div className="wizard-actions">
      {step !== 2 && <button className="wizard-button wizard-button-primary" type="submit" disabled={saving}>
        {saving ? '저장 중…' : step === 1 ? '다음' : editing ? '저장하기' : '시작하기'}
        <span aria-hidden="true">{saving ? '…' : '→'}</span>
      </button>}
      {(step > 1 || editing) && <div className="wizard-secondary-actions">
        {step > 1 && <button className="wizard-button wizard-button-secondary" type="button" disabled={saving}
          onClick={() => { setStep(step - 1); setError(''); }}><span aria-hidden="true">←</span>이전</button>}
        {editing && <button className="wizard-button wizard-button-ghost" type="button" disabled={saving} onClick={onCancel}>취소</button>}
      </div>}
      </div>
    </form>
    </div>
    <p className="storage-note"><span aria-hidden="true">⌁</span>
      프로필은 이 PC의 서버에 저장돼요.<br />브라우저를 닫아도 알림은 동작할 수 있어요.</p>
  </section>;
}
