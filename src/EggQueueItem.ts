import {EggType} from "./EggType.js";
import {CannonSide} from "./CannonSide.js";

export class EggQueueItem {
    public stalled: number;
    public type: EggType;
    public cannon: CannonSide;

    public constructor(stalled: number, type: EggType, cannon: CannonSide) {
        this.stalled = stalled;
        this.type = type;
        this.cannon = cannon;
    }

    public clone(): EggQueueItem {
        return new EggQueueItem(this.stalled, this.type, this.cannon);
    }
}
