import { useEffect, useState } from 'react';
import { getSnapshot } from '../api/activityApi';
import type { ActivitySnapshot, ConnectionStatus } from '../types';

interface Received {
  value: ActivitySnapshot;
  mono: number;
}

export function useActivity(enabled: boolean) {
  const [received, setReceived] = useState<Received | null>(null);
  const [apiReachable, setApiReachable] = useState<boolean | null>(null);
  const [clock, setClock] = useState(() => ({ wall: Date.now(), mono: performance.now() }));

  useEffect(() => {
    if (!enabled) return;
    let active = true;
    let controller: AbortController | null = null;
    let nextPoll: ReturnType<typeof setTimeout> | undefined;
    let deadline: ReturnType<typeof setTimeout> | undefined;
    const tick = () => setClock({ wall: Date.now(), mono: performance.now() });
    const clockTimer = setInterval(tick, 1000);
    async function poll() {
      const request = new AbortController();
      controller = request;
      // Bounds HTTP reachability, not CSI freshness (owned by backend).
      deadline = setTimeout(() => {
        request.abort();
        if (active) setApiReachable(false);
      }, 4000);
      try {
        const value = await getSnapshot(request.signal);
        if (!active || request.signal.aborted) return;
        setReceived({ value, mono: performance.now() });
        setApiReachable(true);
      } catch {
        if (active) setApiReachable(false);
      } finally {
        clearTimeout(deadline);
        if (active) {
          tick();
          nextPoll = setTimeout(() => { void poll(); }, 1000);
        }
      }
    }
    void poll();
    return () => {
      active = false;
      clearInterval(clockTimer);
      clearTimeout(nextPoll);
      clearTimeout(deadline);
      controller?.abort();
    };
  }, [enabled]);

  const value = received?.value.current;
  const streamStatus = value?.connection_status ?? 'CONNECTING';
  const connection: ConnectionStatus = apiReachable === false ? 'RECONNECTING' : streamStatus;
  // Only this hook decides activity validity; components do not infer freshness.
  const current = apiReachable === true && value?.is_live ? value : null;
  const duration = current && received ? Math.min(current.confirmed_duration_seconds,
    current.duration_seconds + Math.max(0, (clock.mono - received.mono) / 1000)) : 0;
  return { current, duration, connection, apiReachable, streamStatus,
    history: received?.value.history ?? [], historyLoaded: received !== null,
    incomplete: connection !== 'LIVE', now: clock.wall };
}
