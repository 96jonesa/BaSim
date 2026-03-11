export class CannonCommand {
    constructor(id, cannon, penance, eggType, numEggs, tick) {
        this.stalled = 0;
        this.id = id;
        this.cannon = cannon;
        this.penance = penance;
        this.eggType = eggType;
        this.numEggs = numEggs;
        this.tick = tick;
    }
    clone() {
        const cmd = new CannonCommand(this.id, this.cannon, this.penance, this.eggType, this.numEggs, this.tick);
        cmd.stalled = this.stalled;
        return cmd;
    }
}
