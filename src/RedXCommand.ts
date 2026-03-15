import {Command} from "./Command.js";

export class RedXCommand extends Command {
    public readonly healerId: number;

    public constructor(healerId: number) {
        super();
        this.healerId = healerId;
    }

    public clone(): RedXCommand {
        return new RedXCommand(this.healerId);
    }
}
