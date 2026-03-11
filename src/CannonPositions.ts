import {Position} from "./Position.js";
import {CannonSide} from "./CannonSide.js";

export const CANNON_RANGE: number = 15;
export const WEST_CANNON_POSITION: Position = new Position(21, 26);
export const EAST_CANNON_POSITION: Position = new Position(40, 26);

export function getCannonPosition(side: CannonSide): Position {
    return side === CannonSide.WEST ? WEST_CANNON_POSITION : EAST_CANNON_POSITION;
}
