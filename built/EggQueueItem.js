export class EggQueueItem {
    constructor(stalled, type, cannon, forcedSplashes = null) {
        this.stalled = stalled;
        this.type = type;
        this.cannon = cannon;
        this.forcedSplashes = forcedSplashes;
    }
    clone() {
        return new EggQueueItem(this.stalled, this.type, this.cannon, this.forcedSplashes);
    }
}
