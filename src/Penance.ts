import {Character} from "./Character.js";
import {Position} from "./Position.js";
import {BarbarianAssault} from "./BarbarianAssault.js";

/**
 * Represents a Barbarian Assault penance character.
 */
export abstract class Penance extends Character {
    public destination: Position;
    public penanceId: number;

    public constructor(position: Position, penanceId: number) {
        super(position);
        this.destination = position;
        this.penanceId = penanceId;
    }

    public abstract tick(barbarianAssault: BarbarianAssault): void;
}