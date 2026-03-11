import {Position} from "./Position.js";
import {Player} from "./Player.js";
import {BarbarianAssault} from "./BarbarianAssault.js";
import {HealerCodeAction} from "./HealerCodeAction.js";
import {HealerPenance} from "./HealerPenance.js";

/**
 * Represents a Barbarian Assault healer player.
 */
export class HealerPlayer extends Player {
    public codeQueue: Array<HealerCodeAction> = [];
    public codeIndex: number = 0;

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
        this.move();
    }

    private processCodeQueue(barbarianAssault: BarbarianAssault): void {
        if (this.codeIndex >= this.codeQueue.length) {
            return;
        }

        const action = this.codeQueue[this.codeIndex];

        // Wait until the specified tick
        if (barbarianAssault.ticks < action.waitUntil) {
            return;
        }

        // Skip dead healers
        const healer = this.findHealer(barbarianAssault, action.healerId);
        if (healer === null) {
            this.codeIndex++;
            return;
        }

        // Check if adjacent to healer
        const adjacent = this.position.closestAdjacentPosition(healer.position);
        if (this.position.equals(adjacent)) {
            healer.eatFood(barbarianAssault);
            this.codeIndex++;
            // Immediately target next action if there is one
            if (this.codeIndex < this.codeQueue.length) {
                const nextAction = this.codeQueue[this.codeIndex];
                if (barbarianAssault.ticks >= nextAction.waitUntil) {
                    const nextHealer = this.findHealer(barbarianAssault, nextAction.healerId);
                    if (nextHealer !== null) {
                        const nextAdj = this.position.closestAdjacentPosition(nextHealer.position);
                        this.findPath(barbarianAssault, nextAdj);
                    }
                }
            }
        } else {
            // Pathfind to adjacent tile of target healer
            this.findPath(barbarianAssault, adjacent);
        }
    }

    private findHealer(barbarianAssault: BarbarianAssault, healerId: number): HealerPenance | null {
        for (const healer of barbarianAssault.healers) {
            if (healer.id === healerId && !healer.isDying) {
                return healer;
            }
        }
        return null;
    }

    /**
     * Creates a deep clone of this object.
     *
     * @return  a deep clone of this object
     */
    public clone(): HealerPlayer {
        let healerPlayer: HealerPlayer = new HealerPlayer(this.position);
        healerPlayer.position = this.position === null ? null : this.position.clone();
        healerPlayer.codeQueue = this.codeQueue.map(a => a.clone());
        healerPlayer.codeIndex = this.codeIndex;

        return healerPlayer;
    }
}
