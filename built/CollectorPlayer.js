import { Player } from "./Player.js";
/**
 * Represents a Barbarian Assault collector player.
 */
export class CollectorPlayer extends Player {
    constructor(position) {
        super(position);
    }
    /**
     * @inheritDoc
     */
    tick(barbarianAssault) {
        if (this.codeQueue.length > 0) {
            this.processCodeQueue(barbarianAssault);
        }
        if (this.arriveDelay) {
            this.arriveDelay = false;
        }
        else {
            this.move();
        }
    }
    /**
     * Creates a deep clone of this object.
     *
     * @return  a deep clone of this object
     */
    clone() {
        let collectorPlayer = new CollectorPlayer(this.position);
        collectorPlayer.position = this.position === null ? null : this.position.clone();
        collectorPlayer.codeQueue = this.codeQueue.map(a => a.clone());
        collectorPlayer.codeIndex = this.codeIndex;
        collectorPlayer.arriveDelay = this.arriveDelay;
        return collectorPlayer;
    }
}
