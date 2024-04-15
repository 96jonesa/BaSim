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
    }

    /**
     * Creates a deep clone of this object.
     *
     * @return  a deep clone of this object
     */
    public clone(): AttackerPlayer {
        let attackerPlayer: AttackerPlayer = new AttackerPlayer(this.position);
        attackerPlayer.position = this.position === null ? null : this.position.clone();

        return attackerPlayer;
    }
}