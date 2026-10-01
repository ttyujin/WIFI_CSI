// Import the existing files without moving/renaming them. Vite includes these
// assets in production builds as well as serving them during development.
import grandmaMoving from '../../images/grandma-moving.gif';
import grandmaStaying from '../../images/grandma-sitaying.gif';
import grandpaMoving from '../../images/grandpa-moving.gif';
import grandpaStaying from '../../images/grandpa-staying.gif';
import type { CharacterType, KnownActivityState } from '../types';

export const IMAGES: Record<CharacterType, Record<KnownActivityState, string>> = {
  grandma: { MOVING: grandmaMoving, STAYING: grandmaStaying },
  grandpa: { MOVING: grandpaMoving, STAYING: grandpaStaying },
};
export const STATE_TEXT = {
  MOVING: { name: '활동 중', description: '움직임이 감지되고 있어요' },
  STAYING: { name: '머무르는 중', description: '큰 움직임이 감지되지 않고 있어요' },
};
