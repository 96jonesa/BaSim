import { Command } from "./Command.js";
export class ToggleRunCommand extends Command {
    constructor() {
        super();
    }
    clone() {
        return new ToggleRunCommand();
    }
}
