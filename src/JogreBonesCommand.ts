import {Command} from "./Command.js";

export class JogreBonesCommand extends Command {
    public constructor() {
        super();
    }

    public clone(): JogreBonesCommand {
        return new JogreBonesCommand();
    }
}
