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
        this.move();
    }
    /**
     * Creates a deep clone of this object.
     *
     * @return  a deep clone of this object
     */
    clone() {
        let collectorPlayer = new CollectorPlayer(this.position);
        collectorPlayer.position = this.position === null ? null : this.position.clone();
        return collectorPlayer;
    }
}
