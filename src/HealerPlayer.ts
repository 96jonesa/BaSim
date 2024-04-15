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
    }

    /**
     * Creates a deep clone of this object.
     *
     * @return  a deep clone of this object
     */
    public clone(): HealerPlayer {
        let healerPlayer: HealerPlayer = new HealerPlayer(this.position);
        healerPlayer.position = this.position === null ? null : this.position.clone();

        return healerPlayer;
    }
}