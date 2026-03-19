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
            this.move(barbarianAssault);
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
        healerPlayer.checkpoints = this.checkpoints.map(p => p.clone());
        healerPlayer.checkpointIndex = this.checkpointIndex;
        healerPlayer.pathDestination = this.pathDestination === null ? null : this.pathDestination.clone();
        healerPlayer.codeQueue = this.codeQueue.map(a => a.clone());
        healerPlayer.codeIndex = this.codeIndex;

        healerPlayer.arriveDelay = this.arriveDelay;
        healerPlayer.prevPosition = this.prevPosition === null ? null : this.prevPosition.clone();
        healerPlayer.isRunning = this.isRunning;
        healerPlayer.pendingSeed = this.pendingSeed;
        healerPlayer.seedMovedThisTick = this.seedMovedThisTick;
        healerPlayer.preSeedPosition = this.preSeedPosition === null ? null : this.preSeedPosition.clone();
        healerPlayer.seedMovedToPosition = this.seedMovedToPosition === null ? null : this.seedMovedToPosition.clone();
        healerPlayer.repeatSeedType = this.repeatSeedType;
        healerPlayer.redXHealerId = this.redXHealerId;
        healerPlayer.isRedXPath = this.isRedXPath;
        healerPlayer.phased = this.phased;

        return healerPlayer;
    }
}
