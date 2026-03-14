import { Player } from "./Player.js";
/**
 * Represents a Barbarian Assault attacker player.
 */
export class AttackerPlayer extends Player {
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
        let attackerPlayer = new AttackerPlayer(this.position);
        attackerPlayer.position = this.position === null ? null : this.position.clone();
        attackerPlayer.checkpoints = this.checkpoints.map(p => p.clone());
        attackerPlayer.checkpointIndex = this.checkpointIndex;
        attackerPlayer.pathDestination = this.pathDestination === null ? null : this.pathDestination.clone();
        attackerPlayer.codeQueue = this.codeQueue.map(a => a.clone());
        attackerPlayer.codeIndex = this.codeIndex;
        attackerPlayer.arriveDelay = this.arriveDelay;
        attackerPlayer.prevPosition = this.prevPosition === null ? null : this.prevPosition.clone();
        attackerPlayer.isRunning = this.isRunning;
        return attackerPlayer;
    }
}
