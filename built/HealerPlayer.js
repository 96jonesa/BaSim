import { Player } from "./Player.js";
/**
 * Represents a Barbarian Assault healer player.
 */
export class HealerPlayer extends Player {
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
        let healerPlayer = new HealerPlayer(this.position);
        healerPlayer.position = this.position === null ? null : this.position.clone();
        healerPlayer.pathQueueIndex = this.pathQueueIndex;
        healerPlayer.pathQueuePositions = [];
        for (let i = 0; i < this.pathQueuePositions.length; i++) {
            healerPlayer.pathQueuePositions.push(this.pathQueuePositions[i] === null ? null : this.pathQueuePositions[i].clone());
        }
        healerPlayer.shortestDistances = [...this.shortestDistances];
        healerPlayer.waypoints = [...this.waypoints];
        healerPlayer.codeQueue = this.codeQueue.map(a => a.clone());
        healerPlayer.codeIndex = this.codeIndex;
        healerPlayer.arriveDelay = this.arriveDelay;
        healerPlayer.prevPosition = this.prevPosition === null ? null : this.prevPosition.clone();
        healerPlayer.isRunning = this.isRunning;
        return healerPlayer;
    }
}
