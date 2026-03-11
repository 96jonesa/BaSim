import {Command} from "./Command.js";

export class HealerCodeCommand extends Command {
    public healerId: number;
    public count: number;

    public constructor(healerId: number, count: number) {
        super();
        this.healerId = healerId;
        this.count = count;
    }

    public clone(): HealerCodeCommand {
        return new HealerCodeCommand(this.healerId, this.count);
    }
}
