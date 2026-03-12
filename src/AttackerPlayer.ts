import {Player} from "./Player.js";
import {Position} from "./Position.js";
import {BarbarianAssault} from "./BarbarianAssault.js";

/**
 * Represents a Barbarian Assault attacker player.
 */
export class AttackerPlayer extends Player {
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
    public clone(): AttackerPlayer {
        let attackerPlayer: AttackerPlayer = new AttackerPlayer(this.position);
        attackerPlayer.position = this.position === null ? null : this.position.clone();
        attackerPlayer.codeQueue = this.codeQueue.map(a => a.clone());
        attackerPlayer.codeIndex = this.codeIndex;
        attackerPlayer.arriveDelay = this.arriveDelay;
        attackerPlayer.prevPosition = this.prevPosition === null ? null : this.prevPosition.clone();

        return attackerPlayer;
    }
}