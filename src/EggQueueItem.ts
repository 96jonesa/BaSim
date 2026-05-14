import {EggType} from "./EggType.js";
import {CannonSide} from "./CannonSide.js";

export class EggQueueItem {
    public stalled: number;
    public type: EggType;
    public cannon: CannonSide;
    public forcedSplashes: number | null;

    public constructor(stalled: number, type: EggType, cannon: CannonSide, forcedSplashes: number | null = null) {
        this.stalled = stalled;
        this.type = type;
        this.cannon = cannon;
        this.forcedSplashes = forcedSplashes;
    }

    public clone(): EggQueueItem {
        return new EggQueueItem(this.stalled, this.type, this.cannon, this.forcedSplashes);
    }
}
