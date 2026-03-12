import {Command} from "./Command.js";

export class WalkRunCommand extends Command {
    public readonly isRunning: boolean;

    public constructor(isRunning: boolean) {
        super();
        this.isRunning = isRunning;
    }

    public clone(): WalkRunCommand {
        return new WalkRunCommand(this.isRunning);
    }
}
