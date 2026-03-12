import {Command} from "./Command.js";

export class ToggleRunCommand extends Command {
    public constructor() {
        super();
    }

    public clone(): ToggleRunCommand {
        return new ToggleRunCommand();
    }
}
