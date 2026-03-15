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
            this.move(barbarianAssault);
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
        collectorPlayer.checkpoints = this.checkpoints.map(p => p.clone());
        collectorPlayer.checkpointIndex = this.checkpointIndex;
        collectorPlayer.pathDestination = this.pathDestination === null ? null : this.pathDestination.clone();
        collectorPlayer.codeQueue = this.codeQueue.map(a => a.clone());
        collectorPlayer.codeIndex = this.codeIndex;
        collectorPlayer.arriveDelay = this.arriveDelay;
        collectorPlayer.prevPosition = this.prevPosition === null ? null : this.prevPosition.clone();
        collectorPlayer.isRunning = this.isRunning;
        collectorPlayer.pendingSeed = this.pendingSeed;
        collectorPlayer.seedMovedThisTick = this.seedMovedThisTick;
        collectorPlayer.preSeedPosition = this.preSeedPosition === null ? null : this.preSeedPosition.clone();
        collectorPlayer.seedMovedToPosition = this.seedMovedToPosition === null ? null : this.seedMovedToPosition.clone();
        collectorPlayer.repeatSeedType = this.repeatSeedType;
        collectorPlayer.redXHealerId = this.redXHealerId;
        collectorPlayer.isRedXPath = this.isRedXPath;
        return collectorPlayer;
    }
}
