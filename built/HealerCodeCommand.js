import { Command } from "./Command.js";
export class HealerCodeCommand extends Command {
    constructor(entries) {
        super();
        this.entries = entries;
    }
    clone() {
        return new HealerCodeCommand(this.entries.map(e => ({ healerId: e.healerId, count: e.count })));
    }
}
