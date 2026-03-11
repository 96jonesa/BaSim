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
        if (this.codeQueue.length > 0) {
            this.processCodeQueue(barbarianAssault);
        }
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
        collectorPlayer.codeQueue = this.codeQueue.map(a => a.clone());
        collectorPlayer.codeIndex = this.codeIndex;

        return collectorPlayer;
    }
}