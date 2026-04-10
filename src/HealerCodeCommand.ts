import {Command} from "./Command.js";

export class HealerCodeCommand extends Command {
    public entries: Array<{ healerId: number; count: number }>;

    public constructor(entries: Array<{ healerId: number; count: number }>) {
        super();
        this.entries = entries;
    }

    public clone(): HealerCodeCommand {
        return new HealerCodeCommand(this.entries.map(e => ({ healerId: e.healerId, count: e.count })));
    }
}
