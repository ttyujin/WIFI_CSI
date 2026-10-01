import type { DailyTotals } from '../types';
import { formatDuration } from '../utils/time';

export function DailySummary({ totals, loaded, incomplete }: {
  totals: DailyTotals; loaded: boolean; incomplete: boolean;
}) {
  return <section className="daily-summary" aria-labelledby="daily-title">
    <div className="section-heading"><p className="eyebrow">오늘 쌓인 시간</p><h2 id="daily-title">오늘 활동 요약</h2></div>
    <div className="summary-grid">
      <div className="summary-tile staying"><p>오늘 머무른 시간</p><strong data-testid="staying-total">{loaded ? formatDuration(totals.STAYING) : '--:--:--'}</strong></div>
      <div className="summary-tile moving"><p>오늘 움직인 시간</p><strong data-testid="moving-total">{loaded ? formatDuration(totals.MOVING) : '--:--:--'}</strong></div>
    </div>
    <p className="summary-note">{incomplete ? '새로운 결과를 기다리는 중이에요. 마지막으로 받은 기록 기준입니다.' :
      '현재 API에 남아 있는 오늘 기록 기준이에요. 서버 재시작 전 기록은 포함되지 않을 수 있어요.'}</p>
  </section>;
}
