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
        this.prevPosition = this.position.clone();
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
        collectorPlayer.pathQueueIndex = this.pathQueueIndex;
        collectorPlayer.pathQueuePositions = [];
        for (let i = 0; i < this.pathQueuePositions.length; i++) {
            collectorPlayer.pathQueuePositions.push(this.pathQueuePositions[i] === null ? null : this.pathQueuePositions[i].clone());
        }
        collectorPlayer.shortestDistances = [...this.shortestDistances];
        collectorPlayer.waypoints = [...this.waypoints];
        collectorPlayer.codeQueue = this.codeQueue.map(a => a.clone());
        collectorPlayer.codeIndex = this.codeIndex;
        collectorPlayer.arriveDelay = this.arriveDelay;
        collectorPlayer.prevPosition = this.prevPosition === null ? null : this.prevPosition.clone();
        collectorPlayer.isRunning = this.isRunning;
        return collectorPlayer;
    }
}
