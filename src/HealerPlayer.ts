import {Position} from "./Position.js";
import {Player} from "./Player.js";
import {BarbarianAssault} from "./BarbarianAssault.js";

/**
 * Represents a Barbarian Assault healer player.
 */
export class HealerPlayer extends Player {

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
        this.prevPosition = this.position.clone();
        if (this.arriveDelay) {
            this.arriveDelay = false;
        } else {
            this.move();
        }
    }

    /**
     * Creates a deep clone of this object.
     *
     * @return  a deep clone of this object
     */
    public clone(): HealerPlayer {
        let healerPlayer: HealerPlayer = new HealerPlayer(this.position);
        healerPlayer.position = this.position === null ? null : this.position.clone();
        healerPlayer.pathQueueIndex = this.pathQueueIndex;
        healerPlayer.pathQueuePositions = [];
        for (let i: number = 0; i < this.pathQueuePositions.length; i++) {
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
