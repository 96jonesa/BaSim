import {CannonSide} from "./CannonSide.js";
import {PenanceType} from "./PenanceType.js";
import {EggType} from "./EggType.js";

export class CannonCommand {
    public id: number;
    public cannon: CannonSide;
    public penance: PenanceType;
    public eggType: EggType;
    public numEggs: number;
    public tick: number;
    public stalled: number = 0;

    public constructor(id: number, cannon: CannonSide, penance: PenanceType, eggType: EggType, numEggs: number, tick: number) {
        this.id = id;
        this.cannon = cannon;
        this.penance = penance;
        this.eggType = eggType;
        this.numEggs = numEggs;
        this.tick = tick;
    }

    public clone(): CannonCommand {
        const cmd = new CannonCommand(this.id, this.cannon, this.penance, this.eggType, this.numEggs, this.tick);
        cmd.stalled = this.stalled;
        return cmd;
    }
}
