export class EggQueueItem {
    constructor(stalled, type, cannon) {
        this.stalled = stalled;
        this.type = type;
        this.cannon = cannon;
    }
    clone() {
        return new EggQueueItem(this.stalled, this.type, this.cannon);
    }
}
