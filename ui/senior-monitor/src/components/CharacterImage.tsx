import { useState } from 'react';

export function CharacterImage({ src, alt }: { src: string; alt: string }) {
  const [failedSrc, setFailedSrc] = useState<string | null>(null);
  return failedSrc === src
    ? <p className="character-placeholder">캐릭터 이미지를 불러오지 못했어요.</p>
    : <img src={src} alt={alt} onError={() => setFailedSrc(src)} />;
}
