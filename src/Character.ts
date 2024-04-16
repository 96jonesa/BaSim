import {Position} from "./Position.js";
import {BarbarianAssault} from "./BarbarianAssault.js";

/**
 * Represents a Barbarian Assault character entity.
 */
export abstract class Character {
    public position: Position;

    protected constructor(position: Position) {
        this.position = position;
    }

    /**
     * Progresses the state of this character by one tick, using the given Barbarian Assault game
     * state as reference.
     *
     * @param barbarianAssault  the game state to refer to when progressing this character's state
     */
    public abstract tick(barbarianAssault: BarbarianAssault): void;

    /**
     * Creates a deep clone of this object.
     *
     * @return  a deep clone of this object
     */
    public abstract clone(): Character;
}