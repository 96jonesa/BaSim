import {Position} from "./Position.js";
import {Player} from "./Player.js";
import {BarbarianAssault} from "./BarbarianAssault.js";

/**
 * Represents a Barbarian Assault collector player.
 */
export class CollectorPlayer extends Player {
    public constructor(position: Position) {
        super(position);
    }

    /**
     * @inheritDoc
     */
    public tick(barbarianAssault: BarbarianAssault): void {
        this.move();
    }

    /**
     * Creates a deep clone of this object.
     *
     * @return  a deep clone of this object
     */
    public clone(): CollectorPlayer {
        let collectorPlayer: CollectorPlayer = new CollectorPlayer(this.position);
        collectorPlayer.position = this.position === null ? null : this.position.clone();

        return collectorPlayer;
    }
}